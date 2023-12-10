from algs.dispersion import Dispersion

def create_alg(alg_name, val_loader, device, args):
    alg_dict = {
        "dispersion": Dispersion
    }
    model = alg_dict[alg_name](val_loader, device, args)
    return model
