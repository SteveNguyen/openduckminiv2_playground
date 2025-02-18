import jax.numpy as jp

contacts = jp.array([0, 1])
ref_foot_contacts = jp.array([1, 0])

contact_rew = jp.sum(contacts == ref_foot_contacts)
print(contact_rew)