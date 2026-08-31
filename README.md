# slop-proxy

Serve Anthropic- and OpenAI-compatible API endpoints backed by OpenAI Codex
subscription (ChatGPT) accounts and Anthropic (Claude Max) accounts. Log in
through the CLI, pool multiple accounts with rotation and failover, issue
per-user API tokens, and track token usage.

Requests for `claude-*` models (configurable via `models.anthropic_patterns`)
are relayed verbatim to the Anthropic API over the pooled Max accounts, sticky
per session so prompt caches keep hitting. Everything else is translated to
the Codex backend. Log in to Max accounts with
`slop-proxy login --provider anthropic`.

Endpoints: `POST /v1/messages`, `POST /v1/chat/completions`, `GET /v1/models`,
`POST /v1/responses` — streaming, tools, images, and reasoning. Requested model
names pass through to the backend as-is; use `slop-proxy models` for the real slugs.

## NixOS module

```nix
{
  inputs.slop-proxy.url = "github:koss/slop-proxy";

  outputs = { nixpkgs, slop-proxy, ... }: {
    nixosConfigurations.host = nixpkgs.lib.nixosSystem {
      modules = [
        slop-proxy.nixosModules.default
        {
          services.slop-proxy = {
            enable = true;
            bind = "[::1]:8484";
          };
        }
      ];
    };
  };
}
```

The service runs as a dedicated `slop-proxy` user with its database at
`/var/lib/slop-proxy/slop.db`. Log in and mint tokens against that database:

```sh
slop-proxy slop-proxy --db /var/lib/slop-proxy/slop.db login
slop-proxy slop-proxy --db /var/lib/slop-proxy/slop.db token create --user alice
```

Point a client at it with the issued token:

```sh
ANTHROPIC_BASE_URL=http://[::1]:8484 ANTHROPIC_API_KEY=sp-... claude
OPENAI_BASE_URL=http://[::1]:8484/v1 OPENAI_API_KEY=sp-...
```

## Per-token limits and metering

Each issued token can carry rolling-window limits. Omitted request or token
limits are unlimited. Token usage counts input plus output tokens as settled
after each request.

```sh
# 60 requests and 100k tokens per hour, each admitted request delayed 250ms
slop-proxy token create --user alice \
  --requests 60 --tokens 100000 --window-seconds 3600 --slowdown-ms 250

slop-proxy token limits 1 --requests 120 --window-seconds 3600
slop-proxy token usage 1
```

Admissions persist before upstream dispatch, so concurrent requests cannot
race past the request limit. A request that takes the token total over its
limit completes, and later requests get `429` until usage rolls out of the
window. Responses carry `x-ratelimit-*` headers, and limit errors include
`retry-after`.
