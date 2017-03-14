#coding=utf-8
from __future__ import division
from numpy import *
from sampleBlock2 import *


def main():
    print "loading sampleblock..."
    sampler = SampleBlock()
    while True:
        sen = raw_input("Please input a sentence:\n")
        print "you input: %s  \n" % (sen)
        k= input("Please input answer num:\n") 
        ans ,align, rester, updater = sampler.getSamples(sen, k)

        for i in range(0, len(ans)):
            print "%d sentence: %s score: %f \n" % (i, ans[i][0], ans[i][1])

        which = input("please input which sentence to show the alignment")

        print ('final align')
        print align[which] 

        for_retend = []
        for_uptend = []


        avg_forward_rester = rester[which]
        avg_forward_updater = updater[which]

        for i in range(0, avg_forward_rester.shape[0]-1):
            for_retend.append(avg_forward_rester[i+1]-avg_forward_rester[i])
        
        for_retend = numpy.array(for_retend)

        for i in range(0, avg_forward_updater.shape[0]-1):
            for_uptend.append(avg_forward_updater[i+1]-avg_forward_updater[i])
        
        for_uptend = numpy.array(for_uptend)


        print (for_retend) / 2.0
        print (for_uptend) / 2.0

if __name__ == "__main__":
    main()
