# ML Copilot — dashboard image.
#
# Three stages: install, build, run. The runtime stage carries Node and the
# compiled application and nothing else — no TypeScript, no ESLint, no Vitest,
# no `node_modules` full of build tooling.
#
# ---------------------------------------------------------------------------
# The one thing worth understanding before changing this file
# ---------------------------------------------------------------------------
# `NEXT_PUBLIC_API_BASE_URL` is inlined into the JavaScript bundle at **build**
# time, because the browser is what reads it. Two consequences follow, and
# both are deliberate:
#
#   1. It is a build argument here, not a runtime environment variable.
#      Changing the backend URL means rebuilding the image — `docker compose
#      up --build`. A runtime value would be simpler to operate but would
#      require the application to fetch its own configuration before it could
#      make its first request, which is a worse trade for a local stack.
#
#   2. It must be a URL **the browser** can resolve. `http://backend:8000` is
#      the Compose service name; it resolves inside the network and nowhere
#      else, so a browser given it would fail every request. Compose therefore
#      passes the published host URL — `http://localhost:8000` by default.
#
# No backend secret is ever passed here. Anything named `NEXT_PUBLIC_` is
# served to every visitor, so only genuinely public values may be build args.

# ---------------------------------------------------------------------------
# Stage 1 — dependencies
# ---------------------------------------------------------------------------
FROM node:22-alpine AS deps

WORKDIR /app

# The lockfile is the point of this stage: `npm ci` installs exactly what it
# pins, and the layer is reused until one of these two files changes.
COPY package.json package-lock.json ./
RUN npm ci

# ---------------------------------------------------------------------------
# Stage 2 — build
# ---------------------------------------------------------------------------
FROM node:22-alpine AS builder

WORKDIR /app

COPY --from=deps /app/node_modules ./node_modules
COPY . .

# Baked into the bundle. Compose supplies the published backend URL; the
# default matches the one in `lib/api/client.ts`, so an image built without a
# build argument still points somewhere sensible rather than nowhere.
ARG NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
ENV NEXT_PUBLIC_API_BASE_URL=${NEXT_PUBLIC_API_BASE_URL}

# Telemetry off: a build should not phone home.
ENV NEXT_TELEMETRY_DISABLED=1

# The same gates the project runs in development, run here so a broken build
# fails at `docker compose build` rather than at first page load.
RUN npm run lint \
    && npm run typecheck \
    && npm run build

# ---------------------------------------------------------------------------
# Stage 3 — runtime
# ---------------------------------------------------------------------------
FROM node:22-alpine AS runtime

ENV NODE_ENV=production \
    NEXT_TELEMETRY_DISABLED=1 \
    HOSTNAME=0.0.0.0 \
    PORT=3000

WORKDIR /app

# `output: "standalone"` in next.config.mjs emits a self-contained server plus
# only the `node_modules` it actually reaches, which is what keeps this stage
# from needing the full dependency tree.
COPY --from=builder /app/public ./public
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static

# The `node` user ships with the base image; nothing here needs root.
USER node

EXPOSE 3000

# Asks the running server for a real page, so a wedged process fails the check.
HEALTHCHECK --interval=15s --timeout=5s --start-period=15s --retries=5 \
    CMD ["node", "-e", "require('http').get({host:'127.0.0.1',port:process.env.PORT||3000,path:'/dashboard'},r=>process.exit(r.statusCode===200?0:1)).on('error',()=>process.exit(1))"]

CMD ["node", "server.js"]
