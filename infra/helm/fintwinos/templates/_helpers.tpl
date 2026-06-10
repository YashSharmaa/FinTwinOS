{{/*
Common template helpers for the FinTwinOS chart.
*/}}

{{/*
Expand the name of the chart.
*/}}
{{- define "fintwinos.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name, truncated to the 63-character DNS
limit. If the release name already contains the chart name it is used as-is.
*/}}
{{- define "fintwinos.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Chart name and version, as used by the chart label.
*/}}
{{- define "fintwinos.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels.
*/}}
{{- define "fintwinos.labels" -}}
helm.sh/chart: {{ include "fintwinos.chart" . }}
{{ include "fintwinos.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/part-of: fintwinos
{{- end }}

{{/*
Selector labels (kept minimal: these are immutable on a Deployment).
*/}}
{{- define "fintwinos.selectorLabels" -}}
app.kubernetes.io/name: {{ include "fintwinos.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}
