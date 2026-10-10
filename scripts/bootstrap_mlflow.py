import os
import yaml
import time
from pathlib import Path

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException


MODEL_NAME = os.getenv(
    "MLFLOW_REGISTERED_MODEL_NAME",
    "FraudGuardFraudDetector",
)
MODEL_PATH = Path(
    os.getenv("BOOTSTRAP_MODEL_PATH", "/app/deployment/model")
)
TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://mlflow:5000",
)
ALIAS = os.getenv("MLFLOW_MODEL_ALIAS", "champion")


def wait_for_mlflow(timeout_seconds: int = 180) -> MlflowClient:
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()
    deadline = time.monotonic() + timeout_seconds

    while time.monotonic() < deadline:
        try:
            client.search_registered_models(max_results=1)
            print(f"MLflow is reachable at {TRACKING_URI}", flush=True)
            return client
        except Exception as exc:
            print(f"Waiting for MLflow: {exc}", flush=True)
            time.sleep(3)

    raise TimeoutError(f"MLflow did not become ready at {TRACKING_URI}")


def main() -> None:
    if not (MODEL_PATH / "MLmodel").is_file():
        raise FileNotFoundError(f"MLmodel metadata missing from {MODEL_PATH}")

    client = wait_for_mlflow()

    try:
        client.get_registered_model(MODEL_NAME)
        print(f"Registered model exists: {MODEL_NAME}", flush=True)
    except MlflowException:
        try:
            client.create_registered_model(MODEL_NAME)
            print(f"Created registered model: {MODEL_NAME}", flush=True)
        except MlflowException:
            # Another bootstrap instance may have created it concurrently.
            client.get_registered_model(MODEL_NAME)

    versions = client.search_model_versions(
        filter_string=f"name = '{MODEL_NAME}'"
    )

    if versions:
        version_number = max(versions, key=lambda v: int(v.version)).version
        print(
            f"Using existing model version {version_number}.",
            flush=True,
        )
    else:
        print(f"Loading packaged model from {MODEL_PATH}", flush=True)
        model = mlflow.sklearn.load_model(str(MODEL_PATH))

        # Log through the MLflow client so the server stores the artifact
        # in its configured artifact store, rather than a local container path.
        with mlflow.start_run(run_name="fraudguard-deployment-bootstrap"):
            metadata = yaml.safe_load(
                (MODEL_PATH / "MLmodel").read_text(encoding="utf-8")
            )
            sklearn_flavor = metadata.get("flavors", {}).get("sklearn", {})
            trusted_types = sklearn_flavor.get("skops_trusted_types", [])
            mlflow.sklearn.log_model(
                sk_model=model,
                name="fraudguard_model",
                registered_model_name=MODEL_NAME,
                code_paths=[str(MODEL_PATH / "code" / "src")],
                skops_trusted_types=trusted_types,
            )

        versions = client.search_model_versions(
            filter_string=f"name = '{MODEL_NAME}'"
        )
        if not versions:
            raise RuntimeError("MLflow did not create a registered model version")

        version_number = max(versions, key=lambda v: int(v.version)).version
        print(f"Created model version {version_number}", flush=True)

    deadline = time.monotonic() + 180

    while time.monotonic() < deadline:
        version = client.get_model_version(MODEL_NAME, version_number)

        if version.status == "READY":
            break

        if version.status == "FAILED":
            raise RuntimeError(
                f"Model registration failed: {version.status_message}"
            )

        time.sleep(3)
    else:
        raise TimeoutError("Model registration did not finish in time")

    client.set_registered_model_alias(
        name=MODEL_NAME,
        alias=ALIAS,
        version=version_number,
    )

    print(
        f"{MODEL_NAME} version {version_number} is READY "
        f"with alias '{ALIAS}'.",
        flush=True,
    )


if __name__ == "__main__":
    main()