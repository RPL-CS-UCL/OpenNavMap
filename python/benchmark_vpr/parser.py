import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="pass --debug to visualize intermediate results",
    )
    parser.add_argument(
        "--out_dir", type=str, default=None, help="path where outputs are saved"
    )
    parser.add_argument(
        "--vpr_models",
        type=str,
        default="cosplace",
        nargs="+",        
        choices=[
            "netvlad",
            "apgem",
            "sfrs",
            "cosplace",
            "convap",
            "mixvpr",
            "eigenplaces",
            "eigenplaces-indoor",
            "anyloc-urban",
            "anyloc-indoor",
            "anyloc-aerial",
            "anyloc-structured",
            "anyloc-unstructured",
            "anyloc-global",
            "salad",
            "salad-indoor",
            "cricavpr",
            "clique-mining",
        ],
        help="_",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default=None,
        choices=[None, "VGG16", "ResNet18", "ResNet50", "ResNet101", "ResNet152"],
        help="_",
    )
    parser.add_argument("--descriptors_dimension", type=int, default=None, help="_")
    parser.add_argument(
        "--vpr_match_models",
        type=str,
        nargs="+",
        default="single_match",
        choices=[
            "single_match",
            "topo_filter",
            "sequence_match",
            "sequence_match_ransac"
        ]
    )
    parser.add_argument(
        "--vpr_match_seq_lens", 
        type=int, 
        nargs="+",
        default=10)
    parser.add_argument(
        "--image_match_models",
        type=str,
        nargs="+",
        default="master",
        choices=[
            "none",
            "loftr",
            "eloftr",
            "se2loftr",
            "aspanformer",
            "matchformer",
            "sift-lg",
            "superpoint-lg",
            "disk-lg",
            "aliked-lg",
            "doghardnet-lg",
            "roma",
            "tiny-roma",
            "dedode",
            "steerers",
            "dedode-kornia",
            "sift-nn",
            "orb-nn",
            "patch2pix",
            "superglue",
            "r2d2",
            "d2net",
            "duster",
            "master",
            "doghardnet-nn",
            "xfeat",
            "xfeat-star",
            "xfeat-lg",
            "dedode-lg",
            "gim-dkm",
            "gim-lg",
            "omniglue",
            "mickey",
            "xfeat-subpx",
            "xfeat-lg-subpx",
            "dedode-subpx",
            "splg-subpx",
            "aliked-subpx",
        ]
    )

    parser.add_argument("--database_folder", type=str, required=True, help="path/to/database")
    parser.add_argument("--queries_folder", type=str, required=True, help="path/to/queries")
    parser.add_argument("--num_workers", type=int, default=1, help="_")
    parser.add_argument(
        "--batch_size", type=int, default=1, help="set to 1 if database images may have different resolution"
    )
    parser.add_argument(
        "--log_dir", type=str, default="default", help="experiment name, output logs will be saved under logs/log_dir"
    )
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="_")
    parser.add_argument(
        "--recall_values",
        type=int,
        nargs="+",
        default=[1, 5, 10, 20],
        help="values for recall (e.g. recall@1, recall@5)",
    )
    parser.add_argument(
        "--no_labels",
        action="store_true",
        help="set to true if you have no labels and just want to "
        "do standard image retrieval given two folders of queries and DB",
    )
    parser.add_argument(
        "--num_preds_to_save", type=int, default=0, help="set != 0 if you want to save predictions for each query"
    )
    parser.add_argument(
        "--save_only_wrong_preds",
        action="store_true",
        help="set to true if you want to save predictions only for " "wrongly predicted queries",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=None,
        nargs="+",
        help="Resizing shape for images (HxW). If a single int is passed, set the"
        "smallest edge of all images to this value, while keeping aspect ratio",
    )
    parser.add_argument(
        "--save_descriptors",
        action="store_true",
        help="set to True if you want to save the descriptors extracted by the model",
    )
    args = parser.parse_args()

    args.use_labels = not args.no_labels
    
    return args

def check_vpr_params(vpr_model, backbone, descriptors_dimension, image_size):
    if vpr_model == "netvlad":
        if backbone not in [None, "VGG16"]:
            raise ValueError("When using NetVLAD the backbone must be None or VGG16")
        if descriptors_dimension not in [None, 4096, 32768]:
            raise ValueError("When using NetVLAD the descriptors_dimension must be one of [None, 4096, 32768]")
        if descriptors_dimension is None:
            descriptors_dimension = 4096

    elif vpr_model == "sfrs":
        if backbone not in [None, "VGG16"]:
            raise ValueError("When using SFRS the backbone must be None or VGG16")
        if descriptors_dimension not in [None, 4096]:
            raise ValueError("When using SFRS the descriptors_dimension must be one of [None, 4096]")
        if descriptors_dimension is None:
            descriptors_dimension = 4096

    elif vpr_model == "cosplace":
        if backbone is None:
            backbone = "ResNet50"
        if descriptors_dimension is None:
            descriptors_dimension = 512
        if backbone == "VGG16" and descriptors_dimension not in [64, 128, 256, 512]:
            raise ValueError("When using CosPlace with VGG16 the descriptors_dimension must be in [64, 128, 256, 512]")
        if backbone == "ResNet18" and descriptors_dimension not in [32, 64, 128, 256, 512]:
            raise ValueError(
                "When using CosPlace with ResNet18 the descriptors_dimension must be in [32, 64, 128, 256, 512]"
            )
        if backbone in ["ResNet50", "ResNet101", "ResNet152"] and descriptors_dimension not in [
            32,
            64,
            128,
            256,
            512,
            1024,
            2048,
        ]:
            raise ValueError(
                f"When using CosPlace with {backbone} the descriptors_dimension must be in [32, 64, 128, 256, 512, 1024, 2048]"
            )

    elif vpr_model == "convap":
        if backbone is None:
            backbone = "ResNet50"
        if descriptors_dimension is None:
            descriptors_dimension = 512
        if backbone not in [None, "ResNet50"]:
            raise ValueError("When using Conv-AP the backbone must be None or ResNet50")
        if descriptors_dimension not in [None, 512, 2048, 4096, 8192]:
            raise ValueError(
                "When using Conv-AP the descriptors_dimension must be one of [None, 512, 2048, 4096, 8192]"
            )

    elif vpr_model == "mixvpr":
        if backbone is None:
            backbone = "ResNet50"
        if descriptors_dimension is None:
            descriptors_dimension = 512
        if backbone not in [None, "ResNet50"]:
            raise ValueError("When using Conv-AP the backbone must be None or ResNet50")
        if descriptors_dimension not in [None, 128, 512, 4096]:
            raise ValueError("When using Conv-AP the descriptors_dimension must be one of [None, 128, 512, 4096]")

    elif vpr_model == "eigenplaces":
        if backbone is None:
            backbone = "ResNet50"
        if descriptors_dimension is None:
            descriptors_dimension = 512
        if backbone == "VGG16" and descriptors_dimension not in [512]:
            raise ValueError("When using EigenPlaces with VGG16 the descriptors_dimension must be in [512]")
        if backbone == "ResNet18" and descriptors_dimension not in [256, 512]:
            raise ValueError("When using EigenPlaces with ResNet18 the descriptors_dimension must be in [256, 512]")
        if backbone in ["ResNet50", "ResNet101", "ResNet152"] and descriptors_dimension not in [
            128,
            256,
            512,
            2048,
        ]:
            raise ValueError(
                f"When using EigenPlaces with {backbone} the descriptors_dimension must be in [128, 256, 512, 2048]"
            )

    elif vpr_model == "eigenplaces-indoor":
        backbone = "ResNet50"
        descriptors_dimension = 2048

    elif vpr_model == "apgem":
        backbone = "Resnet101"
        descriptors_dimension = 2048

    elif vpr_model.startswith("anyloc"):
        backbone = "DINOv2"
        descriptors_dimension = 49152

    elif vpr_model == "salad":
        backbone = "DINOv2"
        descriptors_dimension = 8448

    elif vpr_model == "clique-mining":
        backbone = "DINOv2"
        descriptors_dimension = 8448

    elif vpr_model == "salad-indoor":
        backbone = "Dinov2"
        descriptors_dimension = 8448

    elif vpr_model == "cricavpr":
        backbone = "Dinov2"
        descriptors_dimension = 10752

    if image_size and len(image_size) > 2:
        raise ValueError(
            f"The --image_size parameter can only take up to 2 values, but has received {len(image_size)}."
        )

    return vpr_model, backbone, descriptors_dimension
