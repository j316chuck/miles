import json
from argparse import Namespace

import pytest

from tests.fast.charts.utils import NAMESPACE, RUN_RELEASE_NAME, objects_of_kind, render_run, requires_helm

from miles.utils.external_utils.command_utils.helm_backend import naming
from miles.utils.external_utils.command_utils.helm_backend.launcher import mooncake

INIT_KWARGS = {"master_server_address": "127.0.0.1:50051", "local_hostname": "localhost"}


def _argv(**overrides: object) -> list[str]:
    kwargs = {**INIT_KWARGS, **overrides}
    return [
        "python",
        "train.py",
        "--object-store-backend",
        "mooncake",
        "--mooncake-store-init-kwargs",
        json.dumps(kwargs),
        "--lr",
        "1e-6",
    ]


def _args(*, object_store_backend: str = "mooncake", **overrides: object) -> Namespace:
    return Namespace(
        object_store_backend=object_store_backend,
        mooncake_store_init_kwargs={**INIT_KWARGS, **overrides},
    )


def _rewritten_kwargs(argv: list[str]) -> dict[str, object]:
    return json.loads(argv[argv.index("--mooncake-store-init-kwargs") + 1])


class TestUseOf:
    def test_reads_the_backend_and_the_port_the_run_configured(self):
        """The chart publishes this port, so reading it wrong points every client at a closed socket."""
        use = mooncake.use_of(_args())

        assert use is not None
        assert use.port == 50051

    def test_ignores_a_run_that_names_another_object_store(self):
        """A run on the default backend must not gain a master StatefulSet it never talks to."""
        assert mooncake.use_of(_args(object_store_backend="ray")) is None

    def test_refuses_an_address_that_carries_no_port(self):
        """A bare host would render a Service on port zero, which nothing can dial."""
        with pytest.raises(AssertionError, match="carries no port"):
            mooncake.use_of(_args(master_server_address="127.0.0.1"))

    def test_refuses_a_mooncake_run_that_configured_no_kwargs(self):
        """There is no address to rewrite then, and a pod's loopback master would hang every client."""
        with pytest.raises(AssertionError, match="is missing"):
            mooncake.use_of(Namespace(object_store_backend="mooncake", mooncake_store_init_kwargs=None))


class TestWithClusterMaster:
    def test_points_the_master_address_at_the_in_cluster_service(self):
        """The launcher's own loopback address means nothing inside a pod, which would hang every client."""
        rewritten = mooncake.with_cluster_master(
            _argv(), use=mooncake.use_of(_args()), host="mooncake.myns.svc.cluster.local"
        )

        assert _rewritten_kwargs(rewritten)["master_server_address"] == "mooncake.myns.svc.cluster.local:50051"

    def test_keeps_the_port_the_run_configured(self):
        """The Service publishes the port the values carry, so rewriting the host must not move the port."""
        use = mooncake.use_of(_args(master_server_address="1.2.3.4:60000"))

        rewritten = mooncake.with_cluster_master(_argv(master_server_address="1.2.3.4:60000"), use=use, host="host")

        assert _rewritten_kwargs(rewritten)["master_server_address"] == "host:60000"

    def test_keeps_every_other_init_kwarg(self):
        """The kwargs are rewritten as a whole, and a dropped one changes how the store is built."""
        rewritten = mooncake.with_cluster_master(_argv(), use=mooncake.use_of(_args()), host="host")

        assert _rewritten_kwargs(rewritten)["local_hostname"] == "localhost"

    def test_leaves_the_rest_of_the_argv_untouched(self):
        """Only the address is cluster-specific; every other argument is the experiment itself."""
        rewritten = mooncake.with_cluster_master(_argv(), use=mooncake.use_of(_args()), host="host")

        assert rewritten[:5] == _argv()[:5]
        assert rewritten[-2:] == ["--lr", "1e-6"]

    def test_passes_a_non_mooncake_run_through_unchanged(self):
        """A run that never asked for mooncake has no address to rewrite, and no kwargs to invent."""
        argv = ["python", "train.py", "--lr", "1e-6"]

        assert mooncake.with_cluster_master(argv, use=None, host="host") == argv


@requires_helm
class TestServiceNameCoupling:
    def test_the_host_it_builds_is_the_service_the_chart_renders(self):
        """The pods dial this name; a chart rename would leave the launcher pointing at nothing."""
        objects = render_run("--set", "run.mooncake.enabled=true")
        services = [
            obj for obj in objects_of_kind(objects, "Service") if obj["metadata"]["name"].endswith("mooncake-master")
        ]

        assert len(services) == 1
        assert mooncake.master_service_host(RUN_RELEASE_NAME, NAMESPACE) == (
            f"{services[0]['metadata']['name']}.{NAMESPACE}.svc.cluster.local"
        )

    def test_the_component_name_is_the_one_the_values_carry(self):
        """The launcher names this Service and dials it, so the two must come out of the same call."""
        objects = render_run("--set", "run.mooncake.enabled=true")
        services = [
            obj for obj in objects_of_kind(objects, "Service") if obj["metadata"]["name"].endswith("mooncake-master")
        ]

        assert services[0]["metadata"]["name"] == naming.component_name(RUN_RELEASE_NAME, mooncake.COMPONENT)
