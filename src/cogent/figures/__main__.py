from importlib import import_module
from cogent.figures import arguments

if __name__ == "__main__":
    args = arguments()
    for name in [
        "fig2",
        "fig3",
        "fig4",
        "fig5",
        "fig6",
        "si_s01_mouse_shared_embedding",
        "si_s02_mouse_representative_gene_maps",
        "si_s03_mouse_community_expression",
        "si_s04_mouse_developmental_shifts",
        "si_s05_kidney_condition_embeddings",
        "si_s06_kidney_integrated_vs_sconly",
        "si_s07_kidney_module_overlap",
        "si_s08_kidney_shift_distributions",
        "si_s09_crc_nat_condition_embeddings",
        "si_s10_crc_nat_stromal_genes",
        "si_s11_axolotl_spatial_transition",
    ]:
        print(name, flush=True)
        import_module(f"cogent.figures.{name}").run(
            args.data_dir, args.output_dir / name
        )
