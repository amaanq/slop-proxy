# slop-proxy

Serve Anthropic- and OpenAI-compatible API endpoints backed by OpenAI Codex
subscription (ChatGPT) accounts. Log in through the CLI, pool multiple accounts
with rotation and failover, issue per-user API tokens, and track token usage.

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
            bind = "127.0.0.1:8484";
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
ANTHROPIC_BASE_URL=http://127.0.0.1:8484 ANTHROPIC_API_KEY=sp-... claude
OPENAI_BASE_URL=http://127.0.0.1:8484/v1 OPENAI_API_KEY=sp-...
```

## Per-token limits and metering

Limits are attached to each issued API token and use a rolling window. Omitted
request or token limits are unlimited. Token usage is input plus output tokens,
including cached input tokens reported by the upstream API.

```sh
# Create a key allowing 60 requests and 100,000 tokens per hour. Every admitted
# request is also delayed by 250 ms.
slop-proxy token create --user alice \
  --requests 60 --tokens 100000 --window-seconds 3600 --slowdown-ms 250

# Replace a key's policy, or omit --requests/--tokens to clear those limits.
slop-proxy token limits 1 --requests 120 --window-seconds 3600

# Inspect the current rolling-window meter by token id or displayed prefix.
slop-proxy token usage 1
```

Meter admissions are persisted before upstream dispatch, so concurrent requests
cannot bypass the request limit. Once usage is known, its token count is settled
into the meter. A request that takes the token total over its limit completes,
and later requests receive `429` until usage rolls out of the window. Responses
include `x-ratelimit-*` headers and `retry-after` is included on limit errors.
