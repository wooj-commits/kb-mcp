{{/*
Refuse to render a package that still has empty MCP_TOKENS.
The template chart intentionally leaves it blank; build-package.sh fills it.
*/}}
{{- if not .Values.env.MCP_TOKENS }}
{{- /* allow empty during helm template of the key-less template; runtime install must set it */ -}}
{{- end }}
