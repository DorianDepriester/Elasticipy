from paraqus.abaqus import OdbReader
from paraqus.writers import AsciiWriter


ODB_PATH = "C:/TEMP/2Dplate.odb"  # Path to the ODB
MODEL_NAME = "Model-1"  # Can be chosen freely
INSTANCE_NAMES = ["PART-1-1"]  # Which instances will be exported
STEP_NAME = "Step-1"  # Name of the step that will be exported
FRAME_INDEX = -1  # Export the last frame of the step

reader = OdbReader(odb_path=ODB_PATH,
                   model_name=MODEL_NAME,
                   instance_names=INSTANCE_NAMES,
                   )

# Start configuring the reader instance by specifying field outputs and
# node/element groups that will be exported. These must of course be
# available in the output database.

# Field export requests
reader.add_field_export_request("U", field_position="nodes")
reader.add_field_export_request("PEEQ", field_position="elements")
reader.add_field_export_request("PE", field_position="elements")

# Request some element sets, so you can have a closer look at these
# elements
reader.add_set_export_request("ESID", set_type="elements",
                              instance_name="PART-1-1")
reader.add_set_export_request("ETOP", set_type="elements",
                              instance_name="PART-1-1")

# Create a writer that will write the exported results to a .vtu file
vtu_writer = AsciiWriter("vtk_output_billet", clear_output_dir=True)

# The method read_instances loops over all part instances for one
# point in time, and returns ParaqusModel instances for each of them.
# You put them in a list here, so they can be inspected, but it is more
# memory-efficient to use them one after another in a for loop (see
# following tutorials)
instance_models = list(reader.read_instances(step_name=STEP_NAME,
                                             frame_index=FRAME_INDEX))
