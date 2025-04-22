from pathlib import Path
from approxgnn.generate import (
    generate_random_kernel,
    generate_accelerator_code,
    get_components_and_code,
    generate_mappings,
    generate_configs,
    generate_verilog,
    Accelerator,
)
from approxgnn.utils import load_component_parameters, prepare_pyplot, save_figure
from approxgnn.io_utils import arg_type_check
import numpy as np
from argparse import ArgumentParser
import json
import csv
from zipfile import ZipFile, ZIP_DEFLATED
from collections import defaultdict
import re
import random

from argparse import ArgumentParser

argp = ArgumentParser()

argp_shared = ArgumentParser(add_help=False)

subparsers = argp.add_subparsers(dest="run_type", required=True)

KERNEL_RANDOM = "random"
KERNEL_SMALL_GAUSS = "small_gauss"
KERNEL_LARGE_GAUSS = "large_gauss"
argp_generate: ArgumentParser = subparsers.add_parser("generate", parents=[argp_shared], help="Generate dataset for QoR evaluation.")
argp_generate.add_argument(
    "-w", "--wirings", type=int, default=20, help="Number of wirings."
)
argp_generate.add_argument(
    "-c", "--configs", type=int, default=50, help="Configurations per wiring."
)
argp_generate.add_argument(
    "-o", "--output", type=Path, default=Path("qor_evaluation"), help="Output directory."
)
argp_generate.add_argument(
    "--components", type=Path, help="EvoApproxLib components JSON path."
)
argp_generate.add_argument(
    "-k",
    "--kernel",
    type=lambda x: arg_type_check(
        x, [KERNEL_RANDOM, KERNEL_SMALL_GAUSS, KERNEL_LARGE_GAUSS]
    ),
    default=KERNEL_RANDOM,
    help="Accelerator kernel to generate [random, small_gauss, large_gauss].",
)


argp_filter: ArgumentParser = subparsers.add_parser("filter", parents=[argp_shared], help="Filer dataset to remove accelerators with low variance.")
argp_filter.add_argument("output", type=Path, help="Output directory.")
argp_filter.add_argument("sources", nargs="+", type=Path, help="Source directories.")

argp_join: ArgumentParser = subparsers.add_parser("join", parents=[argp_shared], help="Join QoR and HW results.")
argp_join.add_argument("qor", type=Path, help="QoR and output results CSV file.")
argp_join.add_argument("hw", type=Path, help="HW logs ZIP file to add.")

args = argp.parse_args()


def run_generate():
    if args.components is None:
        args.components = (Path(__file__).parent / "components/components.json").resolve()

    N_FEATURES = 5

    accelerators: dict[str, Accelerator] = {}
    widths: set[int] = set()

    if args.kernel == KERNEL_RANDOM:
        kernel = None
    elif args.kernel == KERNEL_LARGE_GAUSS:
        kernel = np.array(
            [
                (16, 31, 16),
                (31, 67, 31),
                (16, 31, 16),
            ]
        )
    elif args.kernel == KERNEL_SMALL_GAUSS:
        kernel = np.array(
            [
                (1, 2, 1),
                (2, 4, 2),
                (1, 2, 1),
            ]
        )
    else:
        raise ValueError(f"Invalid kernel value: {args.kernel}")

    print("Generating graphs.")

    for _ in range(args.wirings):
        accelerator = generate_random_kernel(
            N_FEATURES,
            minimum=10,
            maximum=20,
            allowed_bitwidths=[8, 12, 16],
            kernel=kernel,
        )

        accelerators[accelerator.id] = accelerator
        widths = widths.union(accelerator.component_bitwidths)

    print("Generating C code.")

    base_path = args.output
    if not (base_path / "src").exists():
        (base_path / "src").mkdir()

    if not (base_path / "bin").exists():
        (base_path / "bin").mkdir()

    accelerator_c_code = generate_accelerator_code(accelerators.values())

    with open(base_path / "src/_wirings.hpp", "w") as f:
        f.write(accelerator_c_code)

    print("Loading components.")

    component_params = load_component_parameters(args.components)
    components, component_c_code = get_components_and_code(
        widths, args.components, component_params
    )

    component_ids = []
    for c in components.values():
        component_ids.extend(c.keys())

    with open(base_path / "src/_components.hpp", "w") as f:
        f.write(component_c_code)

    print("Generating mappings.")

    maps_c = generate_mappings(component_ids, accelerators.keys())

    with open(base_path / "src/configs.cpp", "w") as f:
        f.write(maps_c)

    print("Generating configurations.")

    configs = []
    n_configs_per_wiring = args.configs
    for op_id, accelerator in accelerators.items():
        configs.extend(
            generate_configs(
                accelerator,
                components,
                n_configs_per_wiring,
            )
        )

    with open(base_path / "bin/_eval.tsv", "w") as f:
        f.write("\n".join(configs))

    print("Saving graphs.")

    with ZipFile(base_path / "bin/accelerators.json.zip", "w", ZIP_DEFLATED) as f:
        f.writestr(
            "accelerators.json",
            json.dumps({k: v.serialize() for k, v in accelerators.items()}, indent=2),
        )

    def _name_to_width(name: str):
        if name.startswith("add8"):
            return 8
        elif name.startswith("add12"):
            return 12
        else:
            return 16

    with ZipFile(base_path / "bin/verilogs.zip", "w", ZIP_DEFLATED) as f:
        for config in configs:
            config_items = config.split()
            config_name = config_items[0]
            graph = accelerators[config_items[1]].graph
            assignments = [(x, _name_to_width(x)) for x in config_items[4:]]
            generate_verilog(graph, config_name, assignments)

            f.writestr(
                f"{config_name}.v", generate_verilog(graph, config_name, assignments)
            )


def run_filter():
    qor: dict[str, dict] = {}
    config_data: dict[str, list[str]] = {}
    evals: list[str] = []
    wirings: dict[str, tuple] = {}
    wiring_psnrs: dict[str, list[float]] = defaultdict(list)

    for source in args.sources:
        load_dir: Path = source
        print(f"Loading from: {load_dir}")

        with open(load_dir / "results.csv") as f:
            qor.update(
                {
                    x["config"]: {
                        "wiring": x["wiring"],
                        "psnr": float(x["psnr"]),
                        "ssim": float(x["ssim"]),
                    }
                    for x in csv.DictReader(f)
                }
            )

        with open(load_dir / "results.csv") as f:
            for result in csv.DictReader(f):
                wiring_psnrs[result["wiring"]].append(float(result["psnr"]))

        with open(load_dir / "_eval.tsv") as f:
            config_data.update(
                {
                    words[0]: words
                    for words in (line.split() for line in f)
                    if words[0] in qor
                }
            )

        with open(load_dir / "_eval.tsv") as f:
            evals.extend(f.readlines())

        with ZipFile(str(load_dir / "accelerators.json.zip")) as f:
            wirings.update(json.loads(f.read("accelerators.json").decode()))

    print(f"Total configurations: {len(config_data)}")

    wiring_psnr_std = sorted((np.std(v), k) for k, v in wiring_psnrs.items())

    allowed_wirings = {d[1] for d in wiring_psnr_std if d[0] > 2.5}

    len_valid_wirings = len(
        [d[0] for d in config_data.values() if d[1] in allowed_wirings]
    )
    print(f"Valid configurations: {len_valid_wirings}")

    # -----------------------------------------------------

    out_dir: Path = args.output

    if not out_dir.exists():
        out_dir.mkdir()

    print(f"Merging to: {out_dir}")

    qor_alt = [
        {"config": k, **v} for k, v in qor.items() if v["wiring"] in allowed_wirings
    ]

    with open(out_dir / "results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["wiring", "config", "psnr", "ssim"])
        writer.writeheader()
        writer.writerows(qor_alt)

    wirings_alt = {k: v for k, v in wirings.items() if k in allowed_wirings}

    with ZipFile(str(out_dir / "accelerators.json.zip"), "w", ZIP_DEFLATED) as f:
        f.writestr("accelerators.json", json.dumps(wirings_alt, indent=2))

    with open(out_dir / "_eval.tsv", "w") as out_f:
        for line in evals:
            words = line.split()
            if words[1] in allowed_wirings:
                print(line, end="", file=out_f)

    lengths = [
        len(d) - 4 + random.random() * 0.75
        for d in config_data.values()
        if d[1] in allowed_wirings
    ]
    ys = [float(d["psnr"]) for d in qor.values() if d["wiring"] in allowed_wirings]

    fig, ax = prepare_pyplot()
    ax.scatter(lengths, ys, s=2.0, alpha=0.25, rasterized=True)
    ax.set_xlabel("Component count")
    ax.set_ylabel("PSNR")
    ax.set_xticks([8, 10, 12, 14, 16, 18, 20, 22, 24])
    save_figure(fig, out_dir / "size_vs_psnr.pdf")

    fig, ax = prepare_pyplot(square=True)
    ax.hist(ys, 100)
    ax.set_xlabel("PSNR")
    ax.set_ylabel("Configuration count")
    save_figure(fig, out_dir / "psnr_histogram.pdf")


def run_join():
    R_DELAY = re.compile(r"data arrival time\s+(\d+\.\d+)", re.MULTILINE)
    R_AREA = re.compile(r"Total cell area:\s+(\d+\.\d+)", re.MULTILINE)
    R_POWER = re.compile(r"^Total.+?(\d+\.\d+)\s+(\wW)$", re.MULTILINE)

    def unit_to_number(unit: str):
        if unit == "mW":
            return 1.0
        elif unit == "uW":
            return 0.001
        elif unit == "W":
            return 1000.0
        else:
            raise RuntimeError(f"Invalid unit encountered: {unit}")

    results: dict[str, dict[str, float]] = defaultdict(dict)

    print("Loading HW logs.")

    with ZipFile(args.hw) as f:
        for i, file in enumerate(f.filelist):
            if file.filename.endswith(".log"):
                name = Path(file.filename).parent.name
                log = f.read(file).decode()
                try:
                    if file.filename.endswith("timing.log"):
                        results[name]["delay"] = float(R_DELAY.search(log).group(1))
                    if file.filename.endswith("area.log"):
                        results[name]["area"] = float(R_AREA.search(log).group(1))
                    if file.filename.endswith("power.log"):
                        match = R_POWER.search(log)
                        results[name]["power"] = float(match.group(1)) * unit_to_number(
                            match.group(2)
                        )
                except AttributeError:
                    print(log)
                    break

    print("Loading results.")

    with open(args.qor) as f:
        reader = csv.DictReader(f)
        qor: dict[str, dict[str, float]] = {
            x["config"]: {
                k: (float(v) if k not in ["config", "wiring"] else v)
                for k, v in x.items()
            }
            for x in reader
        }

    for k, v in results.items():
        qor[k]["area"] = v["area"]
        qor[k]["power"] = v["power"]
        qor[k]["delay"] = v["delay"]
    
    print("Updating results file.")

    with open(args.qor, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["wiring", "config", "psnr", "ssim", "area", "power", "delay"]
        )
        writer.writeheader()
        for line in qor.values():
            writer.writerow(line)


if args.run_type == "generate":
    run_generate()
elif args.run_type == "filter":
    run_filter()
elif args.run_type == "join":
    run_join()
