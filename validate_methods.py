from paper_pipeline import run_method_smoke_tests, set_seed, validate_config
import mffn_config as cfg


if __name__ == "__main__":
    set_seed(cfg.RANDOM_SEED)
    validate_config()
    run_method_smoke_tests()
