//! A port of the mustache.js 4.2.0 parser and renderer, as the JavaScript AI SDK
//! calls it: HTML escaping disabled, no partials, values stringified the way
//! JavaScript string concatenation does.

use serde_json::{Map, Number, Value};

/// A template that mustache.js would reject.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{0}")]
pub struct TemplateError(String);

/// Renders `template` against `view` with mustache.js semantics and no HTML escaping.
pub fn render(template: &str, view: &Value) -> Result<String, TemplateError> {
    let tokens = parse(template)?;
    let mut out = String::new();
    let stack = ContextStack {
        view: view.clone(),
        parent: None,
    };
    render_tokens(&tokens, &stack, &mut out);
    Ok(out)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TagType {
    Name,
    Section,
    Inverted,
    Close,
    Partial,
    Unescaped,
    SetDelimiters,
    Comment,
    Curly,
}

#[derive(Debug, Clone)]
enum RawToken {
    Text(char),
    Tag(TagType, String),
}

#[derive(Debug, Clone)]
enum Token {
    Text(String),
    Value(String),
    Section {
        name: String,
        inverted: bool,
        children: Vec<Token>,
    },
}

fn js_is_whitespace(c: char) -> bool {
    matches!(
        c,
        '\u{0009}'..='\u{000D}'
            | '\u{0020}'
            | '\u{00A0}'
            | '\u{1680}'
            | '\u{2000}'..='\u{200A}'
            | '\u{2028}'
            | '\u{2029}'
            | '\u{202F}'
            | '\u{205F}'
            | '\u{3000}'
            | '\u{FEFF}'
    )
}

fn trim_js_start(s: &str) -> &str {
    s.trim_start_matches(js_is_whitespace)
}

fn trim_js_end(s: &str) -> &str {
    s.trim_end_matches(js_is_whitespace)
}

struct Delimiters {
    open: String,
    close: String,
}

impl Delimiters {
    fn compile(spec: &str) -> Result<Self, TemplateError> {
        let mut parts = spec
            .split(js_is_whitespace)
            .filter(|p| !p.is_empty())
            .take(2);
        match (parts.next(), parts.next()) {
            (Some(open), Some(close)) => Ok(Self {
                open: open.to_string(),
                close: close.to_string(),
            }),
            _ => Err(TemplateError(format!("Invalid tags: {spec}"))),
        }
    }
}

struct Scanner<'a> {
    tail: &'a str,
    pos: usize,
}

impl<'a> Scanner<'a> {
    fn advance(&mut self, len: usize) -> &'a str {
        let (head, tail) = self.tail.split_at(len);
        self.tail = tail;
        self.pos += head.chars().count();
        head
    }

    fn scan_whitespace(&mut self) {
        let skipped = self.tail.len() - trim_js_start(self.tail).len();
        self.advance(skipped);
    }

    fn scan_literal(&mut self, literal: &str) -> bool {
        if self.tail.starts_with(literal) {
            self.advance(literal.len());
            true
        } else {
            false
        }
    }

    fn scan_until_literal(&mut self, literal: &str) -> &'a str {
        match self.tail.find(literal) {
            Some(index) => self.advance(index),
            None => self.advance(self.tail.len()),
        }
    }

    fn scan_until_padded(&mut self, literal: &str) -> &'a str {
        match self.tail.find(literal) {
            Some(index) => {
                let value = trim_js_end(&self.tail[..index]);
                self.advance(value.len())
            }
            None => self.advance(self.tail.len()),
        }
    }

    fn scan_padded(&mut self, literal: &str) -> bool {
        let rest = trim_js_start(self.tail);
        if rest.starts_with(literal) {
            let len = self.tail.len() - rest.len() + literal.len();
            self.advance(len);
            true
        } else {
            false
        }
    }

    fn scan_tag_type(&mut self) -> TagType {
        let tag_type = match self.tail.chars().next() {
            Some('#') => TagType::Section,
            Some('^') => TagType::Inverted,
            Some('/') => TagType::Close,
            Some('>') => TagType::Partial,
            Some('{') => TagType::Curly,
            Some('&') => TagType::Unescaped,
            Some('=') => TagType::SetDelimiters,
            Some('!') => TagType::Comment,
            _ => return TagType::Name,
        };
        self.advance(1);
        tag_type
    }
}

struct LineState {
    tokens: Vec<Option<RawToken>>,
    spaces: Vec<usize>,
    has_tag: bool,
    non_space: bool,
}

impl LineState {
    fn push_text(&mut self, chr: char) {
        if js_is_whitespace(chr) {
            self.spaces.push(self.tokens.len());
        } else {
            self.non_space = true;
        }
        self.tokens.push(Some(RawToken::Text(chr)));
        if chr == '\n' {
            self.strip_space();
        }
    }

    fn strip_space(&mut self) {
        if self.has_tag && !self.non_space {
            for index in self.spaces.drain(..) {
                self.tokens[index] = None;
            }
        } else {
            self.spaces.clear();
        }
        self.has_tag = false;
        self.non_space = false;
    }
}

fn parse(template: &str) -> Result<Vec<Token>, TemplateError> {
    if template.is_empty() {
        return Ok(Vec::new());
    }

    let mut delimiters = Delimiters {
        open: "{{".to_string(),
        close: "}}".to_string(),
    };
    let mut line = LineState {
        tokens: Vec::new(),
        spaces: Vec::new(),
        has_tag: false,
        non_space: false,
    };
    let mut sections: Vec<String> = Vec::new();
    let mut scanner = Scanner {
        tail: template,
        pos: 0,
    };

    while !scanner.tail.is_empty() {
        for chr in scanner.scan_until_literal(&delimiters.open).chars() {
            line.push_text(chr);
        }

        let start = scanner.pos;
        if !scanner.scan_literal(&delimiters.open) {
            break;
        }
        scanner.scan_whitespace();
        line.has_tag = true;

        let tag_type = scanner.scan_tag_type();
        scanner.scan_whitespace();

        let value = match tag_type {
            TagType::SetDelimiters => {
                let value = scanner.scan_until_padded("=");
                scanner.scan_padded("=");
                scanner.scan_until_padded(&delimiters.close);
                value
            }
            TagType::Curly => {
                let value = scanner.scan_until_padded(&format!("}}{}", delimiters.close));
                scanner.scan_padded("}");
                scanner.scan_until_padded(&delimiters.close);
                value
            }
            _ => scanner.scan_until_padded(&delimiters.close),
        };

        if !scanner.scan_padded(&delimiters.close) {
            return Err(TemplateError(format!("Unclosed tag at {}", scanner.pos)));
        }

        let tag_type = if tag_type == TagType::Curly {
            TagType::Unescaped
        } else {
            tag_type
        };
        line.tokens
            .push(Some(RawToken::Tag(tag_type, value.to_string())));

        match tag_type {
            TagType::Section | TagType::Inverted => sections.push(value.to_string()),
            TagType::Close => match sections.pop() {
                None => {
                    return Err(TemplateError(format!(
                        "Unopened section \"{value}\" at {start}"
                    )))
                }
                Some(open) if open != value => {
                    return Err(TemplateError(format!(
                        "Unclosed section \"{open}\" at {start}"
                    )))
                }
                Some(_) => {}
            },
            TagType::Name | TagType::Unescaped => line.non_space = true,
            TagType::SetDelimiters => delimiters = Delimiters::compile(value)?,
            TagType::Partial | TagType::Comment | TagType::Curly => {}
        }
    }

    line.strip_space();

    if let Some(open) = sections.pop() {
        return Err(TemplateError(format!(
            "Unclosed section \"{open}\" at {}",
            scanner.pos
        )));
    }

    Ok(nest(line.tokens.into_iter().flatten()))
}

fn nest(tokens: impl Iterator<Item = RawToken>) -> Vec<Token> {
    let mut stack: Vec<(String, bool, Vec<Token>)> = Vec::new();
    let mut root: Vec<Token> = Vec::new();

    fn collector<'a>(
        root: &'a mut Vec<Token>,
        stack: &'a mut [(String, bool, Vec<Token>)],
    ) -> &'a mut Vec<Token> {
        match stack.last_mut() {
            Some((_, _, children)) => children,
            None => root,
        }
    }

    for token in tokens {
        match token {
            RawToken::Text(chr) => {
                let target = collector(&mut root, &mut stack);
                if let Some(Token::Text(text)) = target.last_mut() {
                    text.push(chr);
                } else {
                    target.push(Token::Text(chr.to_string()));
                }
            }
            RawToken::Tag(TagType::Section, name) => stack.push((name, false, Vec::new())),
            RawToken::Tag(TagType::Inverted, name) => stack.push((name, true, Vec::new())),
            RawToken::Tag(TagType::Close, _) => {
                if let Some((name, inverted, children)) = stack.pop() {
                    collector(&mut root, &mut stack).push(Token::Section {
                        name,
                        inverted,
                        children,
                    });
                }
            }
            RawToken::Tag(TagType::Name | TagType::Unescaped, name) => {
                collector(&mut root, &mut stack).push(Token::Value(name));
            }
            RawToken::Tag(
                TagType::Partial | TagType::Comment | TagType::SetDelimiters | TagType::Curly,
                _,
            ) => {}
        }
    }
    root
}

struct ContextStack<'a> {
    view: Value,
    parent: Option<&'a ContextStack<'a>>,
}

impl ContextStack<'_> {
    fn push(&self, view: Value) -> ContextStack<'_> {
        ContextStack {
            view,
            parent: Some(self),
        }
    }

    fn lookup(&self, name: &str) -> Option<Value> {
        if name == "." {
            return Some(self.view.clone());
        }
        let dotted = name.find('.').is_some_and(|index| index > 0);
        let mut context = Some(self);
        while let Some(current) = context {
            let (hit, value) = if dotted {
                lookup_dotted(&current.view, name)
            } else {
                (has_property(&current.view, name), property(&current.view, name))
            };
            if hit {
                return value;
            }
            context = current.parent;
        }
        None
    }
}

fn lookup_dotted(view: &Value, name: &str) -> (bool, Option<Value>) {
    let names: Vec<&str> = name.split('.').collect();
    let mut intermediate = Some(view.clone());
    let mut hit = false;
    for (index, segment) in names.iter().enumerate() {
        let Some(current) = intermediate.as_ref().filter(|v| !v.is_null()) else {
            break;
        };
        if index == names.len() - 1 {
            hit = has_property(current, segment) || primitive_has_own_property(current, segment);
        }
        intermediate = property(current, segment);
    }
    (hit, intermediate)
}

fn array_index(name: &str, len: usize) -> Option<usize> {
    let index: usize = name.parse().ok()?;
    (index.to_string() == name && index < len).then_some(index)
}

fn has_property(value: &Value, name: &str) -> bool {
    match value {
        Value::Object(map) => map.contains_key(name),
        Value::Array(items) => name == "length" || array_index(name, items.len()).is_some(),
        _ => false,
    }
}

fn primitive_has_own_property(value: &Value, name: &str) -> bool {
    match value {
        Value::String(s) => name == "length" || array_index(name, s.encode_utf16().count()).is_some(),
        _ => false,
    }
}

fn property(value: &Value, name: &str) -> Option<Value> {
    match value {
        Value::Object(map) => map.get(name).cloned(),
        Value::Array(items) if name == "length" => Some(Value::from(items.len())),
        Value::Array(items) => array_index(name, items.len()).map(|i| items[i].clone()),
        Value::String(s) if name == "length" => Some(Value::from(s.encode_utf16().count())),
        Value::String(s) => {
            let units: Vec<u16> = s.encode_utf16().collect();
            array_index(name, units.len())
                .map(|i| Value::String(String::from_utf16_lossy(&units[i..=i])))
        }
        _ => None,
    }
}

fn is_truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().is_some_and(|f| f != 0.0 && !f.is_nan()),
        Value::String(s) => !s.is_empty(),
        Value::Array(_) | Value::Object(_) => true,
    }
}

fn render_tokens(tokens: &[Token], context: &ContextStack<'_>, out: &mut String) {
    for token in tokens {
        match token {
            Token::Text(text) => out.push_str(text),
            Token::Value(name) => {
                if let Some(value) = context.lookup(name).filter(|v| !v.is_null()) {
                    out.push_str(&js_to_string(&value));
                }
            }
            Token::Section {
                name,
                inverted: false,
                children,
            } => {
                let Some(value) = context.lookup(name).filter(is_truthy) else {
                    continue;
                };
                match value {
                    Value::Array(items) => {
                        for item in items {
                            render_tokens(children, &context.push(item), out);
                        }
                    }
                    Value::Bool(_) => render_tokens(children, context, out),
                    other => render_tokens(children, &context.push(other), out),
                }
            }
            Token::Section {
                name,
                inverted: true,
                children,
            } => {
                let render = match context.lookup(name) {
                    None => true,
                    Some(Value::Array(items)) => items.is_empty(),
                    Some(value) => !is_truthy(&value),
                };
                if render {
                    render_tokens(children, context, out);
                }
            }
        }
    }
}

/// Converts a JSON value to a string the way JavaScript's `String(value)` would.
pub(crate) fn js_to_string(value: &Value) -> String {
    match value {
        Value::Null => "null".to_string(),
        Value::Bool(b) => b.to_string(),
        Value::Number(n) => js_number_to_string(n),
        Value::String(s) => s.clone(),
        Value::Array(items) => items
            .iter()
            .map(|item| match item {
                Value::Null => String::new(),
                other => js_to_string(other),
            })
            .collect::<Vec<_>>()
            .join(","),
        Value::Object(_) => "[object Object]".to_string(),
    }
}

fn js_number_to_string(n: &Number) -> String {
    if let Some(i) = n.as_i64() {
        return i.to_string();
    }
    if let Some(u) = n.as_u64() {
        return u.to_string();
    }
    js_f64_to_string(n.as_f64().unwrap_or(f64::NAN))
}

fn js_f64_to_string(x: f64) -> String {
    if x.is_nan() {
        return "NaN".to_string();
    }
    if x == 0.0 {
        return "0".to_string();
    }
    if x.is_infinite() {
        return if x > 0.0 { "Infinity" } else { "-Infinity" }.to_string();
    }
    let sign = if x < 0.0 { "-" } else { "" };
    let sci = format!("{:e}", x.abs());
    let (mantissa, exponent) = sci.split_once('e').unwrap_or((&sci, "0"));
    let digits: String = mantissa.chars().filter(|c| *c != '.').collect();
    let exponent: i64 = exponent.parse().unwrap_or(0);
    let k = digits.len() as i64;
    let n = exponent + 1;

    let body = if k <= n && n <= 21 {
        format!("{digits}{}", "0".repeat((n - k) as usize))
    } else if 0 < n && n <= 21 {
        let (int, frac) = digits.split_at(n as usize);
        format!("{int}.{frac}")
    } else if -6 < n && n <= 0 {
        format!("0.{}{digits}", "0".repeat((-n) as usize))
    } else {
        let e = n - 1;
        let e_sign = if e >= 0 { "+" } else { "-" };
        let (first, rest) = digits.split_at(1);
        if rest.is_empty() {
            format!("{first}e{e_sign}{}", e.abs())
        } else {
            format!("{first}.{rest}e{e_sign}{}", e.abs())
        }
    };
    format!("{sign}{body}")
}

/// Builds the view the AI SDK interpolates with: the caller's variables plus `ldctx`.
pub(crate) fn view_with_context(variables: Option<&Map<String, Value>>, ldctx: Value) -> Value {
    let mut view = variables.cloned().unwrap_or_default();
    view.insert("ldctx".to_string(), ldctx);
    Value::Object(view)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn r(template: &str, view: Value) -> String {
        render(template, &view).unwrap()
    }

    #[test]
    fn renders_variables_without_html_escaping() {
        assert_eq!(r("Hello {{name}}", json!({"name": "<b>&\"'"})), "Hello <b>&\"'");
        assert_eq!(r("{{{name}}} {{&name}}", json!({"name": "<i>"})), "<i> <i>");
    }

    #[test]
    fn missing_variables_render_empty() {
        assert_eq!(r("a{{missing}}b{{ldctx.nope}}c", json!({})), "abc");
    }

    #[test]
    fn nested_dotted_lookup() {
        let view = json!({"ldctx": {"key": "user-1", "kind": "user", "address": {"city": "Oakland"}}});
        assert_eq!(r("{{ldctx.key}} in {{ ldctx.address.city }}", view), "user-1 in Oakland");
    }

    #[test]
    fn stringifies_like_javascript() {
        let view = json!({"i": 42, "f": 0.7, "whole": 3.0, "t": true, "n": null, "a": [1, null, "x", [2, 3]], "o": {"k": 1}, "big": 1e21, "tiny": 1.5e-7});
        assert_eq!(
            r("{{i}}|{{f}}|{{whole}}|{{t}}|{{n}}|{{a}}|{{o}}|{{big}}|{{tiny}}", view),
            "42|0.7|3|true||1,,x,2,3|[object Object]|1e+21|1.5e-7"
        );
    }

    #[test]
    fn sections_and_inverted_sections() {
        let view = json!({"items": [{"n": 1}, {"n": 2}], "yes": true, "no": false, "zero": 0, "empty": [], "obj": {"x": "y"}});
        assert_eq!(r("{{#items}}<{{n}}>{{/items}}", view.clone()), "<1><2>");
        assert_eq!(r("{{#yes}}Y{{/yes}}{{#no}}N{{/no}}{{#zero}}Z{{/zero}}", view.clone()), "Y");
        assert_eq!(r("{{^no}}not{{/no}}{{^empty}}none{{/empty}}{{^yes}}x{{/yes}}", view.clone()), "notnone");
        assert_eq!(r("{{#obj}}{{x}}{{/obj}}", view), "y");
    }

    #[test]
    fn standalone_lines_are_removed() {
        let view = json!({"items": ["a", "b"]});
        assert_eq!(r("start\n  {{#items}}\n- {{.}}\n  {{/items}}\nend", view), "start\n- a\n- b\nend");
        assert_eq!(r("x\n{{! comment }}\ny", json!({})), "x\ny");
    }

    #[test]
    fn set_delimiters() {
        assert_eq!(r("{{=<% %>=}}<% name %> {{name}}", json!({"name": "n"})), "n {{name}}");
    }

    #[test]
    fn reports_malformed_templates() {
        assert!(render("{{#a}}x", &json!({})).is_err());
        assert!(render("{{/a}}", &json!({})).is_err());
        assert!(render("{{a", &json!({})).is_err());
    }

    #[test]
    fn lengths_and_indexes() {
        let view = json!({"s": "abc", "a": [5, 6]});
        assert_eq!(r("{{s.length}} {{a.length}} {{a.1}} {{s.0}}", view), "3 2 6 a");
    }
}
