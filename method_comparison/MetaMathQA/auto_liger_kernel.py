def make_liger_kernel(model):
    from transformers.modeling_utils import PreTrainedModel
    from transformers.utils.import_utils import is_liger_kernel_available
    if is_liger_kernel_available():
        from liger_kernel.transformers import _apply_liger_kernel_to_instance

        if isinstance(model, PreTrainedModel):
            # Patch the model with liger kernels. Use the default kernel configurations.
            _apply_liger_kernel_to_instance(model=model)
        elif hasattr(model, "get_base_model") and isinstance(model.get_base_model(), PreTrainedModel):
            # Patch the base model with liger kernels where model is a PeftModel. Use the default kernel configurations.
            _apply_liger_kernel_to_instance(model=model.get_base_model())
        else:
            print(
                "The model is not an instance of PreTrainedModel. No liger kernels will be applied."
            )
    else:
        print(
            "You have set `use_liger_kernel` to `True` but liger-kernel >= 0.3.0 is not available. "
            "Please install it with `pip install liger-kernel`"
        )
    return model