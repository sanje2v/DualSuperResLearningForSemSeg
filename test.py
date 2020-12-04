#from progress.bar import Bar as ProgressBar
#import time

#progressbar1 = ProgressBar('Training', max=100)

#for i in range(0, 100):
#    progressbar1.next()
#    time.sleep(.1)

#    if i % 20 == 0:
#        print()
#        progressbar2 = ProgressBar('Evaluation', max=50)
        
#        for j in range(0, 50):
#            progressbar2.next()
#            time.sleep(.1)
#        progressbar2.finish()
#        print("\033[A")
#progressbar1.finish()

from tqdm.auto import tqdm
import time

bar_format = "{desc}: {percentage:.1f}%|{bar}| {n_fmt}/{total_fmt} {elapsed}<{remaining} {postfix}"

with tqdm(total=100, desc="TRAINING", colour='green', position=0, leave=False, bar_format=bar_format) as p1:
    for i in range(0, 100):
        time.sleep(.1)

        p1.set_postfix({"Postfix: ": str(i)})
        p1.update()
        
        #p1.set_postfix_str("Loss: %d" % i)

        if i % 20 == 0:
            for j in tqdm(range(0, 50), desc="VALIDATION", colour='yellow', position=1, leave=False, bar_format=bar_format):
                time.sleep(.1)

            tqdm.write('')
            tqdm.write('')

        p1.write("Epoch: %d" % i)

#from torch.utils import tensorboard as tb

#w=tb.SummaryWriter('logs/testing')

#w.add_text("INFO", "Information1", global_step=0)
#w.add_text("INFO", "Information2", global_step=0)

#w.close()