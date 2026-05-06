/**
 * Tiny CSP-safe expression compiler for user-defined potential fields Φ(nx, ny).
 *
 * Why not `new Function(src)` or `eval`?
 *   • CSP-hostile (any 'unsafe-eval' lock kills it),
 *   • lets the user pull arbitrary globals (`window.fetch(...)` etc.),
 *   • impossible to bound CPU/recursion.
 *
 * What we accept (whitelist):
 *   • Numbers:     12, 0.5, 1e-3
 *   • Identifiers: nx, ny, x, y, w, h, r, theta, t, pi, e
 *   • Operators:   + - * / ^ (right-assoc), unary -
 *   • Calls:       sin cos tan asin acos atan atan2 exp log sqrt abs
 *                  min max floor ceil round sign hypot pow tanh sinh cosh
 *   • Grouping:    ( ... )
 *
 * Everything else → SyntaxError with a column hint.
 *
 * Compiled output is a pure JS closure over a fixed `Math.*` set — no global
 * lookups at evaluation time. Returns `{ fn }` on success, `{ error }` on fail.
 *
 * Bounds:
 *   • Source length:    1024 chars
 *   • Identifiers/calls limited to the whitelists below
 *   • Parser is recursive-descent — depth bounded by source length
 */

const SOURCE_MAX = 1024;
const VARS = new Set(["nx", "ny", "x", "y", "w", "h", "r", "theta", "t", "pi", "e"]);
const FNS: Record<string, (...args: number[]) => number> = {
  sin: Math.sin, cos: Math.cos, tan: Math.tan,
  asin: Math.asin, acos: Math.acos, atan: Math.atan, atan2: Math.atan2,
  exp: Math.exp, log: Math.log, sqrt: Math.sqrt, abs: Math.abs,
  min: Math.min, max: Math.max,
  floor: Math.floor, ceil: Math.ceil, round: Math.round, sign: Math.sign,
  hypot: Math.hypot, pow: Math.pow,
  tanh: Math.tanh, sinh: Math.sinh, cosh: Math.cosh,
};

export type FieldEnv = {
  nx: number; ny: number;
  x: number; y: number;
  w: number; h: number;
  r: number; theta: number;
  t: number;
};

type Node =
  | { k: "num"; v: number }
  | { k: "var"; n: keyof FieldEnv | "pi" | "e" }
  | { k: "neg"; a: Node }
  | { k: "bin"; op: "+" | "-" | "*" | "/" | "^"; a: Node; b: Node }
  | { k: "call"; n: keyof typeof FNS; args: Node[] };

class Parser {
  private i = 0;
  constructor(private readonly s: string) {}

  // ── Lexing helpers ────────────────────────────────────────────────
  private peek() { return this.s[this.i]; }
  private eof() { return this.i >= this.s.length; }
  private skip() { while (!this.eof() && /\s/.test(this.s[this.i])) this.i++; }
  private err(msg: string): never {
    throw new SyntaxError(`${msg} at column ${this.i + 1}`);
  }

  // ── Grammar ───────────────────────────────────────────────────────
  // expr   := add
  // add    := mul (('+'|'-') mul)*
  // mul    := pow (('*'|'/') pow)*
  // pow    := unary ('^' pow)?              // right-assoc
  // unary  := '-' unary | atom
  // atom   := num | ident ('(' args? ')')? | '(' expr ')'
  parse(): Node {
    this.skip();
    const node = this.add();
    this.skip();
    if (!this.eof()) this.err(`unexpected '${this.peek()}'`);
    return node;
  }

  private add(): Node {
    let a = this.mul();
    this.skip();
    while (this.peek() === "+" || this.peek() === "-") {
      const op = this.s[this.i++] as "+" | "-";
      const b = this.mul();
      a = { k: "bin", op, a, b };
      this.skip();
    }
    return a;
  }

  private mul(): Node {
    let a = this.pow();
    this.skip();
    while (this.peek() === "*" || this.peek() === "/") {
      const op = this.s[this.i++] as "*" | "/";
      const b = this.pow();
      a = { k: "bin", op, a, b };
      this.skip();
    }
    return a;
  }

  private pow(): Node {
    const a = this.unary();
    this.skip();
    if (this.peek() === "^") {
      this.i++;
      const b = this.pow(); // right-assoc
      return { k: "bin", op: "^", a, b };
    }
    return a;
  }

  private unary(): Node {
    this.skip();
    if (this.peek() === "-") { this.i++; return { k: "neg", a: this.unary() }; }
    if (this.peek() === "+") { this.i++; return this.unary(); }
    return this.atom();
  }

  private atom(): Node {
    this.skip();
    const c = this.peek();
    if (c === "(") {
      this.i++;
      const e = this.add();
      this.skip();
      if (this.peek() !== ")") this.err("expected ')'");
      this.i++;
      return e;
    }
    // number
    if (c && (/[0-9.]/.test(c))) {
      const start = this.i;
      while (!this.eof() && /[0-9_.]/.test(this.s[this.i])) this.i++;
      // exponent
      if (!this.eof() && (this.s[this.i] === "e" || this.s[this.i] === "E")) {
        // careful: "e" is also a constant. Only consume as exponent if followed by digit/sign.
        const next = this.s[this.i + 1];
        if (next && (/[0-9+-]/.test(next))) {
          this.i++;
          if (this.s[this.i] === "+" || this.s[this.i] === "-") this.i++;
          while (!this.eof() && /[0-9]/.test(this.s[this.i])) this.i++;
        }
      }
      const v = Number(this.s.slice(start, this.i).replace(/_/g, ""));
      if (!Number.isFinite(v)) this.err("invalid number literal");
      return { k: "num", v };
    }
    // identifier
    if (c && /[a-zA-Z_]/.test(c)) {
      const start = this.i;
      while (!this.eof() && /[a-zA-Z0-9_]/.test(this.s[this.i])) this.i++;
      const name = this.s.slice(start, this.i);
      this.skip();
      if (this.peek() === "(") {
        if (!(name in FNS)) this.err(`unknown function '${name}'`);
        this.i++;
        const args: Node[] = [];
        this.skip();
        if (this.peek() !== ")") {
          args.push(this.add());
          this.skip();
          while (this.peek() === ",") {
            this.i++;
            args.push(this.add());
            this.skip();
          }
        }
        if (this.peek() !== ")") this.err("expected ')'");
        this.i++;
        const fn = FNS[name as keyof typeof FNS];
        if (args.length < fn.length && fn.length <= 2) {
          // soft check: most Math fns are unary or binary; allow variadic min/max/hypot
        }
        return { k: "call", n: name as keyof typeof FNS, args };
      }
      if (!VARS.has(name)) this.err(`unknown identifier '${name}'`);
      return { k: "var", n: name as (keyof FieldEnv | "pi" | "e") };
    }
    this.err(c ? `unexpected '${c}'` : "unexpected end of expression");
  }
}

function evalNode(n: Node, env: FieldEnv): number {
  switch (n.k) {
    case "num": return n.v;
    case "var":
      if (n.n === "pi") return Math.PI;
      if (n.n === "e")  return Math.E;
      return env[n.n];
    case "neg": return -evalNode(n.a, env);
    case "bin": {
      const a = evalNode(n.a, env), b = evalNode(n.b, env);
      switch (n.op) {
        case "+": return a + b;
        case "-": return a - b;
        case "*": return a * b;
        case "/": return a / b;
        case "^": return Math.pow(a, b);
      }
      return 0;
    }
    case "call": {
      const fn = FNS[n.n];
      const args = n.args.map((a) => evalNode(a, env));
      return fn(...args);
    }
  }
}

export type CompileResult =
  | { ok: true; fn: (env: FieldEnv) => number }
  | { ok: false; error: string };

export function compileFieldExpr(src: string): CompileResult {
  if (typeof src !== "string") return { ok: false, error: "expression must be a string" };
  if (src.length > SOURCE_MAX) return { ok: false, error: `expression too long (max ${SOURCE_MAX} chars)` };
  const trimmed = src.trim();
  if (trimmed === "") return { ok: false, error: "expression is empty" };
  let ast: Node;
  try {
    ast = new Parser(trimmed).parse();
  } catch (e) {
    return { ok: false, error: e instanceof Error ? e.message : String(e) };
  }
  // Smoke-test: evaluate at one point so that purely-static issues (e.g. NaN
  // from log(0)) surface immediately rather than mid-frame.
  try {
    const v = evalNode(ast, { nx: 0.1, ny: 0.1, x: 100, y: 100, w: 800, h: 800, r: 0.1414, theta: 0.785, t: 0 });
    if (!Number.isFinite(v)) return { ok: false, error: "expression evaluates to non-finite at sample point" };
  } catch (e) {
    return { ok: false, error: e instanceof Error ? e.message : String(e) };
  }
  return { ok: true, fn: (env) => evalNode(ast, env) };
}
