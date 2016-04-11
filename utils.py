import pandas as pd

def binning(col, num_bins=10, labels=None):
  #Define min and max values:
  minval = col.min() - 1
  maxval = col.max() + 1

  #create list by adding min and max to cut_points
  #break_points = [minval] + cut_points + [maxval]
  interval = (maxval-minval)/num_bins

  break_points = []
  i = minval
  for i in range(num_bins):
    break_points.append(minval + int(i*interval))
  break_points.append(maxval)

  #print break_points
  #if no labels provided, use default labels 0 ... (n-1)
  # if not labels:
  labels = range(num_bins)
  #print labels, len(labels)

  #Binning using cut function of pandas
  colBin = pd.cut(col,bins=break_points,labels=labels,include_lowest=True)
  #print col[colBin.isnull()]
  #print pd.isnull(colBin).any()
  return colBin

# #Binning age:
# cut_points = [90,140,190]
# labels = ["low","medium","high","very high"]
# data["LoanAmount_Bin"] = binning(data["LoanAmount"], cut_points, labels)
# print pd.value_counts(data["LoanAmount_Bin"], sort=False)
