# TODO Uncomment next line
py_tpl_str = """from sqlite3 import connect
import pickle

from rotd_py.system import FluxTag

surf_id = {surf_id}
samp_id = {samp_id}

with connect('../rotd.db', timeout=60) as cursor:
    sql_cmd = 'SELECT * FROM fluxes WHERE surf_id=? AND samp_id=?'
    row = cursor.execute(sql_cmd, ({surf_id}, {samp_id})).fetchall()[-1]

# Transform db data into Flux-type data.
int_flux_tag, srl_flux, surf_id, samp_len, samp_id, status = row
flux_tag = FluxTag(int_flux_tag)
flux = pickle.loads(srl_flux)

if flux_tag == FluxTag.FLUX_TAG:
    flux.run(samp_len, samp_id)
elif flux_tag == FluxTag.SURF_TAG:
    flux.run_surf(samp_len)
elif flux_tag == FluxTag.STOP_TAG:
    pass
else:
    raise ValueError(f"Unable to get correct tag for surface: {surf_id}; sample: {samp_id}.")

srl_flux = pickle.dumps(flux)  # Serialize again

with connect('../rotd.db', timeout=60) as cursor:
    cursor.execute('UPDATE fluxes SET flux=:srl_flux, status=:status '
                   'WHERE surf_id = :surf_id AND samp_id = :samp_id',
                   {{'srl_flux': srl_flux, 'status': 'COMPLETED', 'surf_id': surf_id, 'samp_id': samp_id}})
"""  # TODO Uncomment

# py_tpl_str = ''  # TODO Comment
