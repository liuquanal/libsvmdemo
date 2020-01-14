package com.tdx.lq.libsvmdemo;

import com.alibaba.fastjson.JSON;
import com.tdx.lq.libsvm.*;

import java.io.IOException;

/**
 * Copyright 2019-2020 All Rights Reserved By WWW.TDX.COM.CN
 *
 * <p>Created By Idea
 *
 * @author TDX.LiuQuan 2020/1/15 上午12:31
 */

public class Demo {
    public static void main(String[] args) throws IOException {
        String fileName = "data2";
        String trainFilepath = Demo.class.getResource("/" + fileName + ".txt").getPath();
        String testFilepath = Demo.class.getResource("/" + fileName + "-test.txt").getPath();

        SVMProblem trainProblem = SVMHelper.readProblem(trainFilepath, new SVMParameter());

        System.out.println("train=>" + JSON.toJSONString(trainProblem));

        SVMParameter trainParam = SVMHelper.buildParam();
        trainParam.C = 4;
        String error_msg = SVM.svm_check_parameter(trainProblem, trainParam);

        if (error_msg != null) {
            System.err.print("ERROR: " + error_msg + "\n");
            System.exit(1);
        }

        SVMModel model = SVM.svm_train(trainProblem, trainParam);

        SVMProblem testProblem = SVMHelper.readProblem(testFilepath, trainParam);

        int count = 0;
        int success = 0;
        for (int i = 0; i < testProblem.x.length; ++i) {
            SVMNode[] node = testProblem.x[i];
            double label = SVM.svm_predict(model, node);
            count++;
            if (label - testProblem.y[i] == 0) {
                success++;
                System.out.printf("%d=>labelPredict=%f,label=%f,正确\n", count, label, testProblem.y[i]);
            } else {
                System.out.printf("%d=>labelPredict=%f,label=%f,错误\n", count, label, testProblem.y[i]);
            }
        }
        System.out.printf("正确率为(%d/%d)：%f\n", success, count, success * 1.000 / count);

    }
}
