default_molpro_template = """***, {name}
{options}
geomtyp=xyz
geometry={{
{natom}
header
{geom}
}}

{basis}

{method}

myenergy = energy(1)
---"""