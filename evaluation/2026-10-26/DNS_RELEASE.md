# Free hostname release — 26 October 2026

The preferred public address is **https://oncoedge.duckdns.org/**. DuckDNS's
account dashboard confirmed the hostname and its IPv4 update to `54.253.96.120`.
Both the system resolver and Google's public DNS returned that AWS static IP.
The previous `https://oncoedge.54-253-96-120.sslip.io/` address still works.

The existing Lightsail instance, app image, model weights and caches are retained.
Only the Caddy gateway was recreated to receive this private `.env` setting:

```dotenv
ONCOEDGE_DOMAIN="oncoedge.duckdns.org, oncoedge.54-253-96-120.sslip.io"
```

The existing Caddyfile supports this multiple-address value without a source
change. Its configuration and the production Compose configuration validated
before the gateway was recreated. No DuckDNS account token was copied to the
server or repository. The static IP does not need a dynamic DNS updater.

## Verification

- Both addresses redirected HTTP to their matching HTTPS URL with HTTP 308.
- Both public `/_stcore/health` endpoints returned HTTP 200 and `ok` with normal
  certificate validation enabled.
- The new hostname served a trusted Let's Encrypt certificate for
  `oncoedge.duckdns.org` over TLS 1.3, expiring 24 January 2027. Caddy manages
  subsequent certificate renewals.
- In the internal browser, the new HTTPS address loaded the sample gallery and
  completed real inference for **Cheek · fine markings**: model label OPMD,
  relative text match 94.2%, two regions, 1.3 seconds on the warm model process.
  This is an execution check, not diagnostic validation.
- The app container kept the same ID and start time, remained healthy with zero
  restarts and no OOM kill. Its model process was not restarted for this change.

Evidence: [network and certificates](dns-verification.json),
[runtime state](dns-runtime.json), [browser state](duckdns-live-ui.txt) and
[completed analysis screenshot](duckdns-live.jpg). The photographed sample's
source and license remain recorded in the bundled gallery attribution files.
No additional AWS resources or paid domain were provisioned.

## Rollback

The old address is already active. To remove only the new gateway hostname, set
the server's private `.env` back to:

```dotenv
ONCOEDGE_DOMAIN=oncoedge.54-253-96-120.sslip.io
```

Then, from `/home/ubuntu/OncoEdge-Jetson`, run:

```bash
sudo docker compose -f compose.yaml -f compose.production.yaml config --quiet
sudo docker compose -f compose.yaml -f compose.production.yaml up -d --no-deps caddy
```

The previous environment file is also backed up privately under
`/home/ubuntu/.local/state/oncoedge/`. Do not delete the model/certificate volumes
or recreate the app container for a hostname-only change.
