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

## Gemini keys

Add an unrestricted key with `accounts add-key --provider gemini`. A key
restricted to an HTTP referrer also needs `--referer`.

```sh
slop-proxy accounts add-key \
  --provider gemini \
  --key "$GEMINI_API_KEY" \
  --referer https://conceptcomix.web.app/
```

Referrer-restricted keys use Google's native Gemini surface because its
OpenAI-compatible endpoint drops the referrer before validating the key.

## Experiential keys

Add a gateway key and opt models into its messages endpoint.

```sh
slop-proxy accounts add-key --provider experiential --key "$EXPERIENTIAL_API_KEY"
```

```toml
[models]
experiential_patterns = ["gpt-6-astra"]
```

This integration supports `/v1/messages` only. Experiential models are not
served over `/v1/responses` or `/v1/chat/completions`. Existing provider routes
stay unchanged until you configure `experiential_patterns`.

## Zen egress proxies

Set `zen.proxy_urls` to send only OpenCode Zen traffic through HTTP proxies.
Credentials may be included in each URL. Requests rotate across the configured
egresses. A network failure skips that proxy for 30 seconds, while an anonymous
`429` honors the upstream cooldown and moves the request to another proxy.

```toml
[zen]
proxy_urls = [
  "http://user:password@proxy-one.example:8080",
  "http://user:password@proxy-two.example:8080",
]
```

`zen.proxy_urls_file` reads one URL per line and may be combined with the inline
list. Use the file setting when URLs contain credentials that should stay out
of the config and the Nix store. Configuring either list disables direct Zen
egress.

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
