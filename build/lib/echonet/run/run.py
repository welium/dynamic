from .. import utils
import threading

utils.segmentation.run(
    run_test = True,
    num_epochs = 20
)

t1 = threading.Thread(
    target=utils.segmentation.run,
    kwargs = {'run_test': True,'num_epochs': 20}
    )
t1.start()
t2 = threading.Thread(
    target=utils.video.run,
    kwargs = {'run_test': True,'num_epochs': 20}
    )
t2.start()

t1.join()
t2.join()