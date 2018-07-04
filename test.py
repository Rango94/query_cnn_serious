from data_helper import data_helper

dh=data_helper()

dh.set_batch_size(100)

a,b=dh.next_batch()

print(a.shape,b.shape)