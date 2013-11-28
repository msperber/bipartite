#!/bin/bash

# on macs, wget might have to be installed first: 'brew install wget'
 
wget http://www.cs.princeton.edu/%7Eblei/lda-c/ap.tgz
tar zxf ap.tgz 
mv ap/* .
rmdir ap
rm ap.tgz

