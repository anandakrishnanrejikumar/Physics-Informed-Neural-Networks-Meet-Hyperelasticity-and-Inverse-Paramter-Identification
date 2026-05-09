import Testing_Force_internal_element_wise as tf
import pandas as pd

df = pd.DataFrame(tf.force_over_500000_load_step)
output_file = "force_over_500000_load_step.xlsx"
df.to_excel(output_file, index=False, header=False)


# df = pd.DataFrame(tf.total_force_details_all_quadrature)
# output_file = "total_force_details_all_quadrature.xlsx"
# df.to_excel(output_file, index=False, header=False)

df = pd.DataFrame(tf.force_vector_405_1_500000)
output_file = "force_vector_405_1_500000.xlsx"
df.to_excel(output_file, index=False, header=False)

