import h5py

def show_h5_tree(path):
    def _attrs(obj):
        if obj.attrs:
            for k, v in obj.attrs.items():
                print(f"        @{k} = {v!r}")

    with h5py.File(path, "r") as f:
        def visitor(name, obj):
            indent = "  " * (name.count("/") )
            if isinstance(obj, h5py.Group):
                print(f"{indent}[Group] {name or '/'}")
                _attrs(obj)
            elif isinstance(obj, h5py.Dataset):
                shape = obj.shape
                dtype = obj.dtype
                chunks = obj.chunks
                comp = obj.compression
                print(f"{indent}[Dataset] {name}  shape={shape} dtype={dtype} chunks={chunks} compression={comp}")
                _attrs(obj)
        print(f"File: {path}")
        f.visititems(visitor)

show_h5_tree("/home/joe/RoboTwin/data/handover_block/demo_clean/data/episode0.hdf5")
