from cogent.figures.mouse_module_maps import run_maps


def run(data_dir, output_dir):
    run_maps(data_dir, output_dir, range(8, 15))


if __name__ == "__main__":
    from cogent.figures import arguments

    args = arguments()
    run(args.data_dir, args.output_dir / "si_s03_mouse_community_expression")
