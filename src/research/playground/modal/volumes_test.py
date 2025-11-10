import modal

vol = modal.Volume.from_name("experiments", create_if_missing=True, version=2)


with vol.batch_upload() as batch:
    batch.put_directory(
        "/Users/david/projects/research/data/modal_volumes/experiments",
        "/",
    )
