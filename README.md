# LLM Workflow Engine (LWE) Vertex AI Provider plugin

Vertex AI Provider plugin for [LLM Workflow Engine](https://github.com/llm-workflow-engine/llm-workflow-engine)

Access to [Google Vertex AI](https://cloud.google.com/vertex-ai/docs/generative-ai/learn/models) text/code models.

## Installation

You must configure access to the Vertex AI API in Google Cloud by either:

* Having credentials configured for your environment (gcloud, workload identity, etc...)
* Storing the path to a service account JSON file as the `GOOGLE_APPLICATION_CREDENTIALS` environment variable

### From packages

Install the latest version of this software directly from github with pip:

```bash
pip install git+https://github.com/llm-workflow-engine/lwe-plugin-provider-vertexai
```

### From source (recommended for development)

Install the latest version of this software directly from git:

```bash
git clone https://github.com/llm-workflow-engine/lwe-plugin-provider-vertexai.git
```

Install the development package:

```bash
cd lwe-plugin-provider-vertexai
pip install -e .
```

## Configuration

Add the following to `config.yaml` in your profile:

```yaml
plugins:
  enabled:
    - provider_vertexai
    # Any other plugins you want enabled...
```

## Usage

From a running LWE shell:

```
/provider vertexai
/model model_name text-bison
```
