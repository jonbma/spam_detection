"
Goal: 
1. Preprocess Emails
2. Remove junk words
3. Run analysis on spam and ham (not spam) emails
4. Naive Bayes to detect spam and predict
"

#Preamble
install.packages("tm")
install.packages("SnowballC")
library(tm)
library(SnowballC)


setwd("~/Dropbox/Data Analysis Projects/Princeton Classes/ORF350 Problem Set/Problem Set 3")
top = "~/Dropbox/Data Analysis Projects/Princeton Classes/ORF350 Problem Set/Problem Set 3"
Directories = c("easy_ham","spam")
dirs = paste(top, Directories, sep ="/")

####### Process Emails ############
source("readRawEmail.R")

#Read in mail using email parser
mail = readAllMessages(dirs = dirs)


#Filter our url, numbers, words starting with digits, punctuation marks mixed with letters
list_textBody = numeric(0)
vec.spam = rep(NA,length(mail))
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
#Form matrix with m rows for each word, n colum for each documents
corp_textBody <- Corpus(VectorSource(list_textBody))
cntr <- list(removePunctuation = TRUE, stemming = TRUE, wordLengths = c(3, 20)) #read only words between 3 and 20 length
"""
Get a 36210 words by 3184 documentss
"""
#Calculate how many times word i appears in document j for res[i,j]
res = as.matrix(TermDocumentMatrix(corp_textBody, control = cntr))

#Calculate if word i appears in document j
res2 = res>0


##### Simple Analysis on Spam and Ham Emails #####

ham_len = length(which(vec.spam == FALSE)) 
#There are 2186 non-spam emails

spam_len = length(which(vec.spam == TRUE))
#There are 994 spam emails

total_len = spam_len + ham_len
#Total emails 3184

#Mean Number of Word Count Where The Word Exists
ham_index = which(vec.spam == FALSE)
ham_total_count <- apply(res[,ham_index],1,sum)
ham_freq_count <- apply(res2[,ham_index],1,sum)
ham_mean_count = ham_total_count/ham_freq_count
sort_ham_mean_count = sort(ham_mean_count, decreasing = TRUE)
print(sort_ham_mean_count[1:10])
#HAM: We find that ximian is the most common non=spam email word at 39 , gaim is the second most at 38


#Frequency of Word Count over All HAM words
ham_freq_count <- apply(res2[,ham_index],1,sum)
sort_ham_freq_count <- sort(ham_freq_count, decreasing = TRUE)
perc_ham_freq_count <- sort_ham_freq_count/ham_len
print(perc_ham_freq_count[1:10])
#HAM: 90% of emails have the, 80% have and, 71% have that, which isn't surprise



#Mean Number of Word Count Where The Word Exists
spam_index = which(vec.spam == TRUE)
spam_total_count <- apply(res[,spam_index],1,sum)
spam_freq_count <- apply(res2[,spam_index],1,sum)
spam_mean_count = spam_total_count/spam_freq_count
sort_spam_mean_count = sort(spam_mean_count, decreasing = TRUE)
print(sort_spam_mean_count[1:10])
#SPAM: Highest mean number in spam emails is enekio, atol, blockquot, freak, milk
#suggests spam has a lot of words related to inappropriate material!

#Frequency of Word Count over All HAM words
spam_freq_count <- apply(res2[,spam_index],1,sum)
sort_spam_freq_count <- sort(spam_freq_count, decreasing = TRUE)
perc_spam_freq_count <- sort_spam_freq_count/spam_len
print(perc_spam_freq_count[1:10])
#SPAM: Most common words in all emails is the, this, you, and. Similar to HAM but less frequency

#### Prepare for Naive Bayes assuming Bernoulli Distribution####

#split data into testing and training
set.seed(1)
testingidx = sample(1:ncol(res),100)
#we're getting 100 random documents
trainingidx = 1:ncol(res)
#we're getting all the documents
trainingidx = trainingidx[-testingidx]
#we're removing our testing documents from our training document

#####compute sufficient statistics#####
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

####### Estimate Parameters ###########
#from the sufficient statistics, calculate the estimated parameters, 2 for each word, each type

#total number of spam emails
n1 = numSpam

#total number of ham emails
n0 = length(vec.spam[trainingidx]) -numSpam

#total number of spam documents with word i
vec.wplus = mat.suffStat[,1]

#total number of ham documents with word i
vec.w2plus = mat.suffStat[,2]-mat.suffStat[,1]

#total count of words in spam doc/total number of spam doc with word
vec.lambda1 = (mat.suffCount[,1] - vec.wplus)/vec.wplus 
vec.lambda1[is.nan(vec.lambda1)] = 0

#total count of words in han doc/total number of han doc with word
vec.lambda2 = (mat.suffCount[,2] - mat.suffCount[,1] - vec.w2plus)/vec.w2plus 
vec.lambda2[is.nan(vec.lambda2)] = 0

#total number of spam docs with word i/total number of spam emails 
vec.theta = vec.wplus/n1 

#total number of ham docs with word i/total number of ham emails
vec.theta2 = vec.w2plus/n0 

#percentage of spam emails
eta = n1/(n1+n0) 

#####  Prediction using fitted parameters in Naive Bayes ###
vec.truth = vec.spam[testingidx]
vec.pred = rep(NA,length(vec.truth))
eps = 0.000001
vec.theta[vec.theta<eps] = eps
vec.theta[vec.theta>(1-eps)] = 1-eps
vec.theta2[vec.theta2<eps] = eps
vec.theta2[vec.theta2>(1-eps)] = 1-eps
vec.lambda1[vec.lambda1 < eps] = eps
vec.lambda2[vec.lambda2 < eps] = eps

#Calculate Probability on Testing Set on 1:100 documents using Log Likelihood Poisson Distribution for Naive Bayes
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

#overall prediction accuracy 
sum(abs(vec.truth-vec.pred))

#prediction accuracy for ham emails
sum(abs(vec.truth[vec.truth==0]-vec.pred[vec.truth==0]))

#prediction accuracy for spam emails
sum(abs(vec.truth[vec.truth==1]-vec.pred[vec.truth==1]))
