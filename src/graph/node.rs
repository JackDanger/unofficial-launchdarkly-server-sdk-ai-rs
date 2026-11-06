use crate::config::LdAiAgentConfig;

use super::types::LdGraphEdge;

/// One node of an agent graph: an agent config plus its outgoing edges
/// (JavaScript: `AgentGraphNode`).
#[derive(Debug, Clone)]
pub struct AgentGraphNode {
    key: String,
    config: LdAiAgentConfig,
    edges: Vec<LdGraphEdge>,
}

impl AgentGraphNode {
    /// Creates a node.
    pub fn new(key: impl Into<String>, config: LdAiAgentConfig, edges: Vec<LdGraphEdge>) -> Self {
        Self {
            key: key.into(),
            config,
            edges,
        }
    }

    /// The agent config key identifying this node.
    pub fn get_key(&self) -> &str {
        &self.key
    }

    /// The node's agent config; call `create_tracker()` on it for node-level metrics.
    pub fn get_config(&self) -> &LdAiAgentConfig {
        &self.config
    }

    /// The outgoing edges to this node's children.
    pub fn get_edges(&self) -> &[LdGraphEdge] {
        &self.edges
    }

    /// Whether this node has no outgoing edges.
    pub fn is_terminal(&self) -> bool {
        self.edges.is_empty()
    }
}
