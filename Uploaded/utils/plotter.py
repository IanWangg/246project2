import matplotlib.pyplot as plt



def getPlotName(mode,num_epochs,batch_size,num_hidden,lr,decay_factor,num_workers=1):
    if mode=='ctr':
        plt_name = "ctr_num_epoch_"+str(num_epochs)+"_batch_size_"+str(batch_size)+"_num_hidden_"+str(num_hidden)+"_lr_"+str(lr)+"_lr-dec_"+str(decay_factor)
        return plt_name
    if mode=='fed':
        plt_name = "fed_num_epoch_"+str(num_epochs)+"_batch_size_"+str(batch_size)+"_num_hidden_"+str(num_hidden)+"_lr_"+str(lr)+"_lr-dec_"+str(decay_factor)+"_num_nodes_"+str(num_workers)
        return plt_name



#================================================================================#
# YOUR CODE HERE
### Comments ######
# (i) To compare different schemes, you must run them for same number of expochs for the following code to work.
# (ii) Ensure that the list plt_names is non-empty and you're always appending atleast one scheme (which you want to plot)
# (iii) You can change the name of the legend entries in legend_arr to make the plots more readable.
# (iv) You may also want to change the files names for the saved plots to avoid overwriting (see comments in the plt methods below)
# (v) Refer to the order of arguments passed to the method getPlotName in its definition.
# We have provided logs for the script below, so you can run and check how the plots look like.
#================================================================================#

plt_names = []

plt_names.append(getPlotName('ctr',100,10,20,0.2,0.99))
legend_arr = ['Centralized (Sample)'] 

## SAMPLE comparison plots (for centralized):
# plt_names.append(getPlotName('ctr',100,10,20,0.2,0.99))
# plt_names.append(getPlotName('ctr',100,10,20,0.1,0.99))
# legend_arr = ['lr=0.2','lr=0.1'] 

## SAMPLE comarison plots (for federated):
# plt_names.append(getPlotName('fed',50,2,20,0.2,0.99,5))
# plt_names.append(getPlotName('fed',50,2,20,0.1,0.99,5))
# legend_arr = ['lr=0.2','lr=0.1'] 

# ================================================================ #
# END YOUR CODE HERE
# ================================================================ #


train_lo=[]
train_acc=[]
test_lo=[]
test_acc=[]


for x in range(len(plt_names)):
  train_lo.append([])
  train_acc.append([])
  test_lo.append([])
  test_acc.append([])
  with open("../logs/"+plt_names[x]+"train_loss.txt", "r") as fp:
    for line in fp:
      train_lo[x].append(float(line.strip()))

  with open("../logs/"+plt_names[x]+"train_accu.txt", "r") as fp:
    for line in fp:
      train_acc[x].append(float(line.strip()))      

  with open("../logs/"+plt_names[x]+"test_loss.txt", "r") as fp:
    for line in fp:
      test_lo[x].append(float(line.strip()))

  with open("../logs/"+plt_names[x]+"test_accu.txt", "r") as fp:
    for line in fp:
      test_acc[x].append(float(line.strip()))   



plt.figure()
for i in range(len(plt_names)):
    plt.plot(test_acc[i])
plt.legend(legend_arr,loc=0,fontsize=13)
plt.grid()
plt.title('Test Accuracy performance',fontsize=13)
plt.xlabel('# Epochs',fontsize=13)
plt.ylabel('Test Accuracy',fontsize=13)
plt.savefig('../plots/test_accu.png')   # YOU CAN USE YOUR OWN CUSTOM NAMES TO AVOID OVERWRITING PREVIOUS STORED FILES!


plt.figure()
for i in range(len(plt_names)):
    plt.plot(train_acc[i])
plt.legend(legend_arr,loc=0,fontsize=13)
plt.grid()
plt.title('Training Accuracy performance',fontsize=13)
plt.xlabel('# Epochs',fontsize=13)
plt.ylabel('Train Accuracy',fontsize=13)
plt.savefig('../plots/train_accu.png')   # YOU CAN USE YOUR OWN CUSTOM NAMES TO AVOID OVERWRITING PREVIOUS STORED FILES!


plt.figure()
for i in range(len(plt_names)):
    plt.plot(train_lo[i])
plt.legend(legend_arr,loc=0,fontsize=13)
plt.grid()
plt.title('Training Loss',fontsize=13)
plt.xlabel('# Epochs',fontsize=13)
plt.ylabel('Loss',fontsize=13)
plt.savefig('../plots/train_loss.png')   # YOU CAN USE YOUR OWN CUSTOM NAMES TO AVOID OVERWRITING PREVIOUS STORED FILES!


plt.figure()
for i in range(len(plt_names)):
    plt.plot(test_lo[i])
plt.legend(legend_arr,loc=0,fontsize=13)
plt.grid()
plt.title('Test Loss',fontsize=13)
plt.xlabel('# Epochs',fontsize=13)
plt.ylabel('Loss',fontsize=13)
plt.savefig('../plots/test_loss.png')   # YOU CAN USE YOUR OWN CUSTOM NAMES TO AVOID OVERWRITING PREVIOUS STORED FILES!
