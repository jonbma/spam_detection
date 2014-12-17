"
Goal: Preprocess and then use Naive Bayes to detect spam!
"

#Preamble
install.packages("tm")
install.packages("SnowballC")
library(tm)
library(SnowballC)
setwd("//Files/jbma/Documents/ORF350 Problem Set/Problem Set 3")
top = "//Files/jbma/Documents/ORF350 Problem Set/Problem Set 3"
Directories = c("easy_ham","spam")
dirs = paste(top, Directories, sep ="/")

source("readRawEmail.R")
mail = readAllMessages(dirs = dirs)

#Part 1: Remove words starting with digits or punctuation marks mixed with letter
list_textBody = numeric(0)
vec.spam = rep(NA,length(mail))

#Regular Expression 
for(i in 1:length(mail)){
  tmpmail = mail[[i]]
  tmp = tmpmail$body
  tmp2 = paste(tmp$text,collapse="")
  tmp3 = gsub("\\b([[:punct:]|[:digit:]])*[a-zA-Z]*([[:punct:]|[:digit:]])+[a-zA-Z]*
              ([[:punct:]|[:digit:]])*"," ",tmp2)
  tmp4 = gsub("[^A-Za-z]"," ",tmp3)
  list_textBody[[i]] = tmp4
  vec.spam[i] = tmpmail$spam
}

numHam = sum(!vec.spam)

corp_textBody <- Corpus(VectorSource(list_textBody))
cntr <- list(removePunctuation = TRUE, stemming = TRUE, wordLengths = c(3, 20))

res = as.matrix(TermDocumentMatrix(corp_textBody, control = cntr))
res2 = res>0

####################Calculate Spam, Ham, and if Email has greater than 1 junk word##############################

ham_len = length(which(vec.spam == FALSE)) #2186
spam_len = length(which(vec.spam == TRUE)) #994
total_len = spam_len + ham_len #3184

#HAM
ham_index = which(vec.spam == FALSE)
#Quantity 1: Mean Number of Word Count Where The Word Exists
#Numerator
ham_total_count <- apply(res[,ham_index],1,sum)

#Denominator
ham_freq_count <- apply(res2[,ham_index],1,sum)

#Result
ham_mean_count = ham_total_count/ham_freq_count
sort_ham_mean_count = sort(ham_mean_count, decreasing = TRUE)
print(sort_ham_mean_count[1:10])

#Quantity 2: Frequency of Word Count over All HAM words

#Numerator
ham_freq_count <- apply(res2[,ham_index],1,sum)
sort_ham_freq_count <- sort(ham_freq_count, decreasing = TRUE)

#Result
perc_ham_freq_count <- sort_ham_freq_count/ham_len
print(perc_ham_freq_count[1:10])

#SPAM
spam_index = which(vec.spam == TRUE)

#Quantity 1: Mean Number of Word Count Where The Word Exists
#Numeratorres[]
spam_total_count <- apply(res[,spam_index],1,sum)

#Denominator
spam_freq_count <- apply(res2[,spam_index],1,sum)

#Result
spam_mean_count = spam_total_count/spam_freq_count
sort_spam_mean_count = sort(spam_mean_count, decreasing = TRUE)
print(sort_spam_mean_count[1:10])

#Quantity 2: Frequency of Word Count over All HAM words
#Numerator
spam_freq_count <- apply(res2[,spam_index],1,sum)
sort_spam_freq_count <- sort(spam_freq_count, decreasing = TRUE)

#Result
perc_spam_freq_count <- sort_spam_freq_count/spam_len
print(perc_spam_freq_count[1:10])

#split data into testing and training
set.seed(1)
testingidx = sample(1:ncol(res),100) #we're getting 100 random documents
trainingidx = 1:ncol(res) #we're getting all the documents
trainingidx = trainingidx[-testingidx] #we're removing our testing documents from our training document

################compute sufficient statistics#####################
#for each word: indicator how many documents use this word, how many spam documents use this word
computeSufficient <- function(vec){
  spamIndic = sum(vec[trainingidx]%*%vec.spam[trainingidx])
  totalIndic = sum(vec[trainingidx])
  return(c(spamIndic,totalIndic))
}
mat.suffStat = apply(res2,1,computeSufficient)
mat.suffStat = t(mat.suffStat)
mat.suffCount = apply(res,1,computeSufficient)
mat.suffCount = t(mat.suffCount)

numSpam = sum(vec.spam[trainingidx])

###########Estimate Parameters ##################################
#from the sufficient statistics, calculate the estimated parameters, 2 for each word, each type

n1 = numSpam #total number of spam emails
n0 = length(vec.spam[trainingidx]) -numSpam #total number of ham emails
vec.wplus = mat.suffStat[,1] #total number of spam documents with word i
vec.w2plus = mat.suffStat[,2]-mat.suffStat[,1] #total number of ham documents with word i

vec.lambda1 = (mat.suffCount[,1] - vec.wplus)/vec.wplus #total count of words in spam doc/total number of spam doc with word
vec.lambda1[is.nan(vec.lambda1)] = 0
vec.lambda2 = (mat.suffCount[,2] - mat.suffCount[,1] - vec.w2plus)/vec.w2plus #total count of words in han doc/total number of han doc with word
vec.lambda2[is.nan(vec.lambda2)] = 0
vec.theta = vec.wplus/n1 #total number of spam docs with word i/total number of spam emails 
vec.theta2 = vec.w2plus/n0 #total number of ham docs with word i/total number of ham emails
eta = n1/(n1+n0) #percentage of spam emails

#prediction
vec.truth = vec.spam[testingidx]
vec.pred = rep(NA,length(vec.truth))
eps = 0.000001
vec.theta[vec.theta<eps] = eps
vec.theta[vec.theta>(1-eps)] = 1-eps
vec.theta2[vec.theta2<eps] = eps
vec.theta2[vec.theta2>(1-eps)] = 1-eps
vec.lambda1[vec.lambda1 < eps] = eps
vec.lambda2[vec.lambda2 < eps] = eps

#Calculate Probability on "Testing" Set on 1:100 documents
#Reminder: res[i,j] means word i in document j

for(j in 1:length(vec.truth))
{
  #print(res2[,testingidx[i]]%*%(log(vec.theta)-vec.lambda1 +res_y[,testingidx[i]]%*%log(vec.lambda1) - log(factorial(res_y[,testingidx[i]] - 1))))
  print(j)

  #Log likelihood for spam
  logprobspam = (eta)
  for(i in 1:(dim(res)[1]))
  {
    #w_i,j = 0, so word doesn't exist
    if(res2[i,testingidx[j]] == 0)
    {
      logprobspam = logprobspam + log(1-vec.theta[i])
    }
    #w_i,j = 1, so word does exist
    else
    {
      logprobspam = logprobspam + log(vec.theta[i]) - vec.lambda1[i] + (res[i,testingidx[j]] - 1)*log(vec.lambda1[i]) - lfactorial(res[i,testingidx[j]]-1)
    }
  }
  
  logprobham = (1-eta)
  for(i in 1:(dim(res)[1]))
  {
    #w_i,j = 0, so word doesn't exist
    if(res2[i,testingidx[j]] == 0)
    {
      logprobham = logprobham + log(1-vec.theta2[i])
    }
    #w_i,j = 1, so word does exist
    else
    {
      logprobham = logprobham + log(vec.theta2[i]) - vec.lambda2[i] + (res[i,testingidx[j]] - 1)*log(vec.lambda2[i]) - lfactorial(res[i,testingidx[j]]-1)
    }
  }
  
  if(logprobspam>logprobham){
    vec.pred[j] = 1
  }
  else {
    vec.pred[j] = 0
  }
}


#Calculate Prediction
print(vec.pred)
sum(abs(vec.truth-vec.pred)) #overall prediction accuracy 
sum(abs(vec.truth[vec.truth==0]-vec.pred[vec.truth==0])) #prediction accuracy for ham emails
sum(abs(vec.truth[vec.truth==1]-vec.pred[vec.truth==1])) #prediction accuracy for spam emails
