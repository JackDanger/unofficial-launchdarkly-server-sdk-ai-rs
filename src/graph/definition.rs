use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::Arc;

use indexmap::{IndexMap, IndexSet};

use crate::config::LdAiAgentConfig;

use super::node::AgentGraphNode;
use super::tracker::LdGraphTracker;
use super::types::LdAgentGraphFlagValue;

/// The context a traversal callback receives: the caller's initial context plus the
/// results of exactly the node's dependencies, keyed by config key.
pub type ExecutionContext<T> = IndexMap<String, T>;

#[derive(Clone)]
struct GraphTrackerFactory(Arc<dyn Fn() -> LdGraphTracker + Send + Sync>);

struct Inner {
    agent_graph: LdAgentGraphFlagValue,
    nodes: IndexMap<String, AgentGraphNode>,
    enabled: bool,
    create_tracker: GraphTrackerFactory,
}

/// An agent graph configuration with its pre-built nodes (JavaScript:
/// `AgentGraphDefinition`). Obtain one from [`crate::LdAiClient::agent_graph`]; when
/// the graph is disabled or invalid, [`Self::enabled`] is `false` and it has no nodes.
/// Clones share the same graph.
#[derive(Clone)]
pub struct AgentGraphDefinition {
    inner: Arc<Inner>,
}

impl fmt::Debug for AgentGraphDefinition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AgentGraphDefinition")
            .field("graph", &self.inner.agent_graph)
            .field("nodes", &self.inner.nodes.keys().collect::<Vec<_>>())
            .field("enabled", &self.inner.enabled)
            .finish()
    }
}

impl AgentGraphDefinition {
    /// Creates a definition. `create_tracker` mints a graph tracker per run.
    pub fn new(
        agent_graph: LdAgentGraphFlagValue,
        nodes: IndexMap<String, AgentGraphNode>,
        enabled: bool,
        create_tracker: impl Fn() -> LdGraphTracker + Send + Sync + 'static,
    ) -> Self {
        Self {
            inner: Arc::new(Inner {
                agent_graph,
                nodes,
                enabled,
                create_tracker: GraphTrackerFactory(Arc::new(create_tracker)),
            }),
        }
    }

    /// Whether the graph is enabled and valid.
    pub fn enabled(&self) -> bool {
        self.inner.enabled
    }

    /// Builds a node for every key in the graph that has an agent config.
    pub fn build_nodes(
        graph: &LdAgentGraphFlagValue,
        agent_configs: &HashMap<String, LdAiAgentConfig>,
    ) -> IndexMap<String, AgentGraphNode> {
        Self::collect_all_keys(graph)
            .into_iter()
            .filter_map(|key| {
                let config = agent_configs.get(&key)?.clone();
                let edges = graph.edges_of(&key).to_vec();
                Some((key.clone(), AgentGraphNode::new(key, config, edges)))
            })
            .collect()
    }

    /// Every node key referenced by the graph: the root, then each edge source and
    /// its targets, in declaration order.
    pub fn collect_all_keys(graph: &LdAgentGraphFlagValue) -> IndexSet<String> {
        let mut keys = IndexSet::new();
        keys.insert(graph.root.clone());
        if let Some(edges) = &graph.edges {
            for (source, targets) in edges {
                keys.insert(source.clone());
                for edge in targets {
                    keys.insert(edge.key.clone());
                }
            }
        }
        keys
    }

    /// The children of `node_key`.
    pub fn get_child_nodes(&self, node_key: &str) -> Vec<&AgentGraphNode> {
        let Some(node) = self.inner.nodes.get(node_key) else {
            return Vec::new();
        };
        node.get_edges()
            .iter()
            .filter_map(|edge| self.inner.nodes.get(&edge.key))
            .collect()
    }

    /// Every node with a direct edge to `node_key`.
    pub fn get_parent_nodes(&self, node_key: &str) -> Vec<&AgentGraphNode> {
        self.inner
            .nodes
            .values()
            .filter(|node| node.get_edges().iter().any(|edge| edge.key == node_key))
            .collect()
    }

    /// Every node with no outgoing edges.
    pub fn terminal_nodes(&self) -> Vec<&AgentGraphNode> {
        self.inner
            .nodes
            .values()
            .filter(|n| n.is_terminal())
            .collect()
    }

    /// The root node; `None` for a disabled graph.
    pub fn root_node(&self) -> Option<&AgentGraphNode> {
        self.inner.nodes.get(&self.inner.agent_graph.root)
    }

    /// The node with `node_key`, if any.
    pub fn get_node(&self, node_key: &str) -> Option<&AgentGraphNode> {
        self.inner.nodes.get(node_key)
    }

    /// All nodes, in graph-declaration order.
    pub fn nodes(&self) -> &IndexMap<String, AgentGraphNode> {
        &self.inner.nodes
    }

    /// The raw graph flag value.
    pub fn get_config(&self) -> &LdAgentGraphFlagValue {
        &self.inner.agent_graph
    }

    /// A new graph tracker with a fresh `runId`; call once per graph run.
    pub fn create_tracker(&self) -> LdGraphTracker {
        (self.inner.create_tracker.0)()
    }

    fn node(&self, key: &str) -> &AgentGraphNode {
        &self.inner.nodes[key]
    }

    fn reachable_and_discovery(&self, root_key: &str) -> (HashSet<String>, Vec<String>) {
        let mut reachable = HashSet::from([root_key.to_string()]);
        let mut order = vec![root_key.to_string()];
        let mut queue = std::collections::VecDeque::from([root_key.to_string()]);
        while let Some(key) = queue.pop_front() {
            let Some(node) = self.inner.nodes.get(&key) else {
                continue;
            };
            for edge in node.get_edges() {
                if self.inner.nodes.contains_key(&edge.key) && reachable.insert(edge.key.clone()) {
                    order.push(edge.key.clone());
                    queue.push_back(edge.key.clone());
                }
            }
        }
        (reachable, order)
    }

    fn scoped<T: Clone>(
        initial: &ExecutionContext<T>,
        deps: &IndexSet<String>,
        results: &HashMap<String, T>,
    ) -> ExecutionContext<T> {
        let mut context = initial.clone();
        for key in deps {
            if let Some(result) = results.get(key) {
                context.insert(key.clone(), result.clone());
            }
        }
        context
    }

    fn pick_next(
        order: &[String],
        visited: &HashSet<String>,
        degree: &HashMap<String, i64>,
        skip: Option<&str>,
    ) -> Option<String> {
        let candidates = || {
            order
                .iter()
                .filter(|k| Some(k.as_str()) != skip && !visited.contains(*k))
        };
        candidates()
            .find(|k| degree[*k] == 0)
            .or_else(|| candidates().min_by_key(|k| degree[*k]))
            .cloned()
    }

    /// Visits every node reachable from the root in topological order
    /// (predecessors first, root first). Ties go to graph-discovery order (BFS from
    /// the root following declared edge order); cycles are broken by lowest remaining
    /// in-degree, so each reachable node is visited exactly once.
    ///
    /// `f` receives the node and a fresh context holding `initial_execution_context`
    /// plus the results of exactly that node's reachable, already-visited ancestors.
    /// Its return value is stored under the node's key.
    pub fn traverse<T: Clone>(
        &self,
        mut f: impl FnMut(&AgentGraphNode, ExecutionContext<T>) -> T,
        initial_execution_context: ExecutionContext<T>,
    ) {
        let Some(root) = self.root_node() else {
            return;
        };
        let root_key = root.get_key().to_string();
        let (reachable, order) = self.reachable_and_discovery(&root_key);

        let mut indeg: HashMap<String, i64> = reachable.iter().map(|k| (k.clone(), 0)).collect();
        for key in &order {
            for edge in self.node(key).get_edges() {
                if let Some(d) = indeg.get_mut(&edge.key) {
                    *d += 1;
                }
            }
        }
        indeg.insert(root_key, 0);

        let mut visited: HashSet<String> = HashSet::new();
        let mut results: HashMap<String, T> = HashMap::new();
        let mut ancestors: HashMap<String, IndexSet<String>> = HashMap::new();

        while visited.len() < reachable.len() {
            let Some(next) = Self::pick_next(&order, &visited, &indeg, None) else {
                break;
            };

            let mut anc = IndexSet::new();
            for parent in self.get_parent_nodes(&next) {
                let parent_key = parent.get_key();
                if !visited.contains(parent_key) {
                    continue;
                }
                anc.insert(parent_key.to_string());
                if let Some(parent_ancestors) = ancestors.get(parent_key) {
                    anc.extend(parent_ancestors.iter().cloned());
                }
            }
            visited.insert(next.clone());

            let context = Self::scoped(&initial_execution_context, &anc, &results);
            ancestors.insert(next.clone(), anc);
            let result = f(self.node(&next), context);
            results.insert(next.clone(), result);

            for edge in self.node(&next).get_edges() {
                if let Some(d) = indeg.get_mut(&edge.key) {
                    *d -= 1;
                }
            }
        }
    }

    /// Visits every node reachable from the root in reverse topological order
    /// (descendants first, root always last). Ties go to graph-discovery order; cycles
    /// are broken by lowest remaining out-degree, so each reachable node is visited
    /// exactly once, even in graphs with no terminal nodes.
    ///
    /// `f` receives the node and a fresh context holding `initial_execution_context`
    /// plus the results of exactly that node's reachable, already-visited descendants;
    /// the root receives every other reachable node's result.
    pub fn reverse_traverse<T: Clone>(
        &self,
        mut f: impl FnMut(&AgentGraphNode, ExecutionContext<T>) -> T,
        initial_execution_context: ExecutionContext<T>,
    ) {
        let Some(root) = self.root_node() else {
            return;
        };
        let root_key = root.get_key().to_string();
        let (reachable, order) = self.reachable_and_discovery(&root_key);

        let mut outdeg: HashMap<String, i64> = order
            .iter()
            .map(|k| {
                let count = self
                    .node(k)
                    .get_edges()
                    .iter()
                    .filter(|e| reachable.contains(&e.key))
                    .count();
                (k.clone(), count as i64)
            })
            .collect();

        let mut visited: HashSet<String> = HashSet::new();
        let mut results: HashMap<String, T> = HashMap::new();
        let mut descendants: HashMap<String, IndexSet<String>> = HashMap::new();

        while order.iter().any(|k| *k != root_key && !visited.contains(k)) {
            let Some(next) = Self::pick_next(&order, &visited, &outdeg, Some(&root_key)) else {
                break;
            };

            let mut desc = IndexSet::new();
            for edge in self.node(&next).get_edges() {
                if !reachable.contains(&edge.key) || !visited.contains(&edge.key) {
                    continue;
                }
                desc.insert(edge.key.clone());
                if let Some(child_descendants) = descendants.get(&edge.key) {
                    desc.extend(child_descendants.iter().cloned());
                }
            }
            visited.insert(next.clone());

            let context = Self::scoped(&initial_execution_context, &desc, &results);
            descendants.insert(next.clone(), desc);
            let result = f(self.node(&next), context);
            results.insert(next.clone(), result);

            let parents: Vec<String> = self
                .get_parent_nodes(&next)
                .iter()
                .map(|p| p.get_key().to_string())
                .collect();
            for parent in parents {
                if parent != root_key && reachable.contains(&parent) {
                    if let Some(d) = outdeg.get_mut(&parent) {
                        *d -= 1;
                    }
                }
            }
        }

        let root_deps: IndexSet<String> =
            order.iter().filter(|k| **k != root_key).cloned().collect();
        let context = Self::scoped(&initial_execution_context, &root_deps, &results);
        f(root, context);
    }
}
