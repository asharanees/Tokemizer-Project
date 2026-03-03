# syntax=docker/dockerfile:1.4
# Multi-stage build for optimized React frontend with nginx
# Requires Docker BuildKit: DOCKER_BUILDKIT=1 docker build .

# ============================================================================
# Stage 1: Builder - Build React application with Vite
# ============================================================================
FROM node:20-alpine AS builder

# Set working directory
WORKDIR /app

# Copy package files for dependency installation
COPY package.json package-lock.json ./

# Install dependencies with cache mount for npm
# This dramatically speeds up rebuilds by caching downloaded packages
RUN --mount=type=cache,target=/root/.npm,sharing=locked \
    npm ci --prefer-offline --no-audit

# Copy application source
COPY . .

# Build the React application
# Vite will output to /app/dist by default
RUN npm run build

# ============================================================================
# Stage 2: Runtime - Serve with nginx (unprivileged for security)
# ============================================================================
FROM nginxinc/nginx-unprivileged:1.27-alpine

# Switch to root to perform cleanup and configuration
USER root

# Remove default nginx content
RUN rm -rf /usr/share/nginx/html/*

# Copy built React app from builder stage
COPY --from=builder /app/dist /usr/share/nginx/html

# Ensure the nginx user owns the files
RUN chown -R nginx:nginx /usr/share/nginx/html

# Create nginx configuration for React SPA with API proxy
# Using environment variables for backend proxy timeout and max upload size.
ARG BACKEND_PROXY_TIMEOUT=900s
ARG MODEL_UPLOAD_MAX_BODY=512m
ENV BACKEND_PROXY_TIMEOUT=${BACKEND_PROXY_TIMEOUT}
ENV MODEL_UPLOAD_MAX_BODY=${MODEL_UPLOAD_MAX_BODY}

# Create nginx configuration
RUN cat > /etc/nginx/conf.d/default.conf <<'EOF'
# Nginx configuration for Tokemizer frontend
# Serves React SPA and proxies API requests to backend

# Define backend upstream
upstream backend {
    server backend:8000;
    keepalive 32;
}

server {
    listen 8080;
    server_name _;
    client_max_body_size ${MODEL_UPLOAD_MAX_BODY};
    access_log off;
    error_log /dev/stderr warn;
    
    # Serve React app
    root /usr/share/nginx/html;
    index index.html;
    
    # Enable gzip compression for better performance
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_types text/plain text/css text/xml text/javascript application/javascript application/json application/xml+rss image/svg+xml;
    
    # Security headers
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Referrer-Policy "strict-origin-when-cross-origin" always;
    
    # API proxy configuration with extended timeouts for long-running optimizations
    location /api/ {
        proxy_pass http://backend;
        proxy_http_version 1.1;
        
        # Headers for proper proxying
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Keep-alive and connection management
        proxy_set_header Connection "";
        proxy_buffering off;
        
        # Extended timeouts for long-running ML optimizations
        # Can be overridden via BACKEND_PROXY_TIMEOUT environment variable
        proxy_read_timeout ${BACKEND_PROXY_TIMEOUT};
        proxy_send_timeout ${BACKEND_PROXY_TIMEOUT};
        proxy_connect_timeout 60s;
        
        # Disable timeout for SSE/streaming if needed
        proxy_request_buffering off;
    }
    
    # React SPA routing - serve index.html for all non-API routes
    location / {
        try_files $uri $uri/ /index.html;
        
        # Cache static assets
        location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg|woff|woff2|ttf|eot)$ {
            expires 1y;
            add_header Cache-Control "public, immutable";
        }
    }
    
    # Health check endpoint
    location /health {
        return 200 "healthy\n";
        add_header Content-Type text/plain;
    }
}
EOF

# Use the official envsubst approach in docker-entrypoint.d
RUN echo '#!/bin/sh' > /docker-entrypoint.d/40-envsubst-on-nginx-conf.sh && \
    echo 'envsubst "\$BACKEND_PROXY_TIMEOUT \$MODEL_UPLOAD_MAX_BODY" < /etc/nginx/conf.d/default.conf > /etc/nginx/conf.d/default.conf.tmp' >> /docker-entrypoint.d/40-envsubst-on-nginx-conf.sh && \
    echo 'mv /etc/nginx/conf.d/default.conf.tmp /etc/nginx/conf.d/default.conf' >> /docker-entrypoint.d/40-envsubst-on-nginx-conf.sh && \
    chmod +x /docker-entrypoint.d/40-envsubst-on-nginx-conf.sh

# Ensure nginx user can modify the configuration at runtime (for envsubst)
RUN chown nginx:nginx /etc/nginx/conf.d/default.conf

# Run as unprivileged user (EXPOSE is just metadata)
USER 101
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD wget --quiet --tries=1 --spider http://localhost:8080/health || exit 1

# Default command
CMD ["nginx", "-g", "daemon off;"]
