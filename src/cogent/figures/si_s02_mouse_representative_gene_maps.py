from cogent.figures.mouse_module_maps import run_maps


def run(data_dir, output_dir):
    run_maps(data_dir, output_dir, range(8))


if __name__ == "__main__":
    from cogent.figures import arguments

    args = arguments()
    run(args.data_dir, args.output_dir / "si_s02_mouse_representative_gene_maps")
