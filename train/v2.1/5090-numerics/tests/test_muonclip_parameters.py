from axolotl.muonclip import (
    get_param_metadata,
    tag_parameters_for_muon,
    tag_parameters_for_optimizer_split,
)


class FakeParam:
    def __init__(self, shape, requires_grad=True, name=""):
        self.shape = tuple(shape)
        self.ndim = len(shape)
        self.requires_grad = requires_grad
        self.name = name
        self.muon_exempt = False


class FakeModule:
    def __init__(self):
        self.params = {
            "attn.q_proj.weight": FakeParam((64, 64), True, "q"),
            "attn.q_proj.bias": FakeParam((64,), True, "q_bias"),
            "attn.k_proj.weight": FakeParam((64, 64), True, "k"),
            "attn.k_proj.bias": FakeParam((64,), True, "k_bias"),
            "norm.weight": FakeParam((64,), True, "ln"),
            "frozen.weight": FakeParam((64, 64), False, "frozen"),
        }

    def named_parameters(self):
        return list(self.params.items())


def test_tag_parameters_sets_attribute_and_counts():
    module = FakeModule()
    summary = tag_parameters_for_muon(module)

    assert summary.total == 6
    assert summary.muon == 2  # q/k weights only
    assert summary.adam == 3  # q/k bias + norm
    assert summary.frozen == 1

    assert getattr(module.params["attn.q_proj.weight"], "use_muon") is True
    assert getattr(module.params["attn.k_proj.weight"], "use_muon") is True
    assert getattr(module.params["attn.q_proj.bias"], "use_muon") is False


def test_tag_parameters_returns_optimizer_splits():
    module = FakeModule()
    module.params["attn.k_proj.weight"].muon_exempt = True

    summary, muon_params, adam_params = tag_parameters_for_optimizer_split(module)
    assert summary.muon == 1  # only q_proj.weight remains
    assert module.params["attn.k_proj.weight"] in adam_params
    assert module.params["attn.q_proj.weight"] in muon_params
    assert module.params["attn.k_proj.weight"] not in muon_params


def test_metadata_fallback_available_for_non_attr_params():
    module = FakeModule()

    # Use an object that forbids attribute assignment to exercise metadata store.
    class NoAttrParam(FakeParam):
        __slots__ = ()

    module.params["blocked.weight"] = NoAttrParam((32, 32))

    summary = tag_parameters_for_muon(module)
    assert summary.total == 7

    blocked = module.params["blocked.weight"]
    assert not hasattr(blocked, "use_muon")
    assert get_param_metadata(blocked, "use_muon") is True
