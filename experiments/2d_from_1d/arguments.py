import argparse


def add_arguments_data(parser):
    parser.add_argument(
        "--data",
        type=str,
        default="two-spirals",
        choices=[
            "eight-gaussians",
            "galaxy",
            "gaussian",
            "hollow",
            "kv",
            "leaf",
            "pinwheel",
            "rings",
            "swissroll",
            "two-spirals",
            "waterbag",
        ],
    )
    parser.add_argument("--data-decorr", type=int, default=0)
    parser.add_argument("--data-noise", type=float, default=None)
    parser.add_argument("--data-size", type=int, default=int(1.00e+06))
    parser.add_argument("--data-warp", type=int, default=0)
    parser.add_argument("--meas-num", type=int, default=6)
    parser.add_argument("--meas-angle-min", type=int, default=0.0)
    parser.add_argument("--meas-angle-max", type=int, default=180.0)
    parser.add_argument("--meas-bins", type=int, default=75)
    parser.add_argument("--meas-noise", type=float, default=0.0)
    parser.add_argument("--meas-noise-type", type=str, default="gaussian")
    parser.add_argument("--meas-xmax", type=float, default=3.0)
    return parser


def add_arguments_train(parser, model="nsf"):
    if model == "ment":
        # mentflow.models.ment.MENT
        parser.add_argument("--epochs", type=int, default=10)
        parser.add_argument("--omega", type=float, default=0.25)
    else:
        # mentflow.core.MENTFlow
        parser.add_argument("--epochs", type=int, default=20)
        parser.add_argument("--iters", type=int, default=300)
        parser.add_argument("--batch-size", type=int, default=30000)
        
        parser.add_argument("--penalty", type=float, default=0.0)
        parser.add_argument("--penalty-step", type=float, default=20.0)
        parser.add_argument("--penalty-scale", type=float, default=1.1)
        parser.add_argument("--penalty-max", type=float, default=None)
        
        parser.add_argument("--rtol", type=float, default=0.0)
        parser.add_argument("--atol", type=float, default=0.0)
        parser.add_argument("--dmax", type=float, default=7.50e-04)
    
        parser.add_argument("--lr", type=float, default=0.005)
        parser.add_argument("--lr-min", type=float, default=0.001)
        parser.add_argument("--lr-drop", type=float, default=0.1)
        parser.add_argument("--lr-patience", type=int, default=400)

        if model == "nn":
            parser.set_defaults(
                lr=0.01,
                penalty=1000.0,
                penalty_step=0.0,
                penalty_scale=1.0,
            )
    return parser


def add_arguments_eval(parser, model="nsf"):
    parser.add_argument("--eval-freq", type=int, default=None)
    parser.add_argument("--vis-freq", type=int, default=None)
    parser.add_argument("--vis-size", type=int, default=int(1.00e+06))
    parser.add_argument("--vis-bins", type=int, default=125)
    parser.add_argument("--vis-line", type=str, default="line", choices=["line", "step"])
    parser.add_argument("--vis-maxcols", type=int, default=7)
    parser.add_argument("--fig-dpi", type=float, default=300)
    parser.add_argument("--fig-ext", type=str, default="png")
    return parser
    

def add_arguments_model(parser, model="nsf"):
    
    if model == "nsf":
        ## zuko.flows.NSF
        parser.add_argument("--transforms", type=int, default=5)
        parser.add_argument("--hidden-units", type=int, default=64)
        parser.add_argument("--hidden-layers", type=int, default=3)
        parser.add_argument("--spline-bins", type=int, default=20)
        parser.add_argument("--perm", type=int, default=1)
        parser.add_argument("--prior-scale", type=float, default=1.0)
        parser.add_argument("--disc", type=str, default="kld", choices=["kld", "mae", "mse"])
        
    elif model == "nn":
        ## mentflow.models.nn.NNGenerator
        parser.add_argument("--input-features", type=int, default=2)
        parser.add_argument("--hidden-units", type=int, default=20)
        parser.add_argument("--hidden-layers", type=int, default=2)
        parser.add_argument("--activation", type=str, default="tanh")
        parser.add_argument("--dropout", type=float, default=0.0)
        parser.add_argument("--prior-scale", type=float, default=1.0)
        parser.add_argument("--entest", type=str, default="cov")
        parser.add_argument("--disc", type=str, default="mae", choices=["kld", "mae", "mse"])
        
    elif model == "ment":
        ## mentflow.models.ment.MENT
        parser.add_argument("--method", type=str, default="integrate")
        parser.add_argument("--interpolate", type=str, default="linear", choices=["nearest", "linear", "pchip"])
        parser.add_argument("--prior", type=str, default="gaussian", choices=["gaussian", "uniform"])
        parser.add_argument("--prior-scale", type=float, default=1.0)
        parser.add_argument("--disc", type=str, default="kld", choices=["kld", "mae", "mse"])
        
    else:
        raise ValueError
        
    return parser
    

def make_parser(model="nsf"):
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=None)
    
    parser = add_arguments_data(parser)
    parser = add_arguments_train(parser, model)
    parser = add_arguments_eval(parser, model)   
    parser = add_arguments_model(parser, model)
    
    return parser

    
    