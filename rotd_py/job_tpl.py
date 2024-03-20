
py_tpl_str = """from sqlite3 import connect
import pickle
import os
import time

from rotd_py.system import FluxTag

surf_id = {surf_id}
face_id = {face_id}
samp_id = {samp_id}

here = os.path.dirname(os.path.abspath(__file__))
i = 0
#Wait for the creation of the pickle file
while not os.path.isfile(os.path.join(here, f'surf{surf_id}_face{face_id}_samp{samp_id}.pkl')):
    time.sleep(5)
    i += 1
    if i > 3:
        exit()
try:
    with open(os.path.join(here, f'surf{surf_id}_face{face_id}_samp{samp_id}.pkl'), 'rb') as pkl_file:
        pickle_job = pickle.load(pkl_file)
except EOFError:
    self.logger.debug(f'Unsuccesful opening of Surface_{surf_id}/jobs/surf{surf_id}_face{face_id}_samp{samp_id}.pkl, retrying...')

# Import serialised job from rotdPy workflow.
int_flux_tag, flux, surf_id, face_id, samp_len, samp_id, status = pickle_job
flux_tag = FluxTag(int_flux_tag)

if flux_tag == FluxTag.FLUX_TAG:
    flux.run(samp_len, face_id, samp_id)
elif flux_tag == FluxTag.SURF_TAG:
    flux.run_surf(samp_len)
elif flux_tag == FluxTag.STOP_TAG:
    pass
else:
    raise ValueError(f"Unable to get correct tag for surface: {surf_id}; face_id: {face_id}; sample: {samp_id}.")

with open(os.path.join(here, f'surf{surf_id}_face{face_id}_samp{samp_id}.pkl'), 'wb') as pkl_file:
    pickle.dump([flux_tag.value, flux, surf_id, face_id, samp_len, samp_id, 'COMPLETED'], pkl_file)
"""  
