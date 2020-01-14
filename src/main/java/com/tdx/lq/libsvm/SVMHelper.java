package com.tdx.lq.libsvm;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.StringTokenizer;
import java.util.Vector;

/**
 * Copyright 2019-2020 All Rights Reserved By WWW.TDX.COM.CN
 *
 * <p>Created By Idea
 *
 * @author TDX.LiuQuan 2020/1/15 上午12:32
 */

public class SVMHelper {
    private static SVMPrintInterface svm_print_null = new SVMPrintInterface() {
        public void print(String s) {
            System.out.println(s);
        }
    };

    public static SVMProblem readProblem(String filepath, SVMParameter param) throws IOException {
        BufferedReader fp = new BufferedReader(new FileReader(filepath));
        Vector<Double> vy = new Vector<Double>();
        Vector<SVMNode[]> vx = new Vector<SVMNode[]>();
        int max_index = 0;

        while (true) {
            String line = fp.readLine();
            if (line == null) break;

            StringTokenizer st = new StringTokenizer(line, " \t\n\r\f:");

            vy.addElement(Double.parseDouble(st.nextToken()));
            int m = st.countTokens() / 2;
            SVMNode[] x = new SVMNode[m];
            for (int j = 0; j < m; j++) {
                x[j] = new SVMNode();
                x[j].index = Integer.parseInt(st.nextToken());
                x[j].value = Double.parseDouble(st.nextToken());
            }
            if (m > 0) max_index = Math.max(max_index, x[m - 1].index);
            vx.addElement(x);
        }

        SVMProblem prob = new SVMProblem();
        prob.l = vy.size();
        prob.x = new SVMNode[prob.l][];
        for (int i = 0; i < prob.l; i++)
            prob.x[i] = vx.elementAt(i);
        prob.y = new double[prob.l];
        for (int i = 0; i < prob.l; i++)
            prob.y[i] = vy.elementAt(i);

        if (param.gamma == 0 && max_index > 0)
            param.gamma = 1.0 / max_index;

        if (param.kernel_type == SVMParameter.PRECOMPUTED)
            for (int i = 0; i < prob.l; i++) {
                if (prob.x[i][0].index != 0) {
                    throw new RuntimeException("Wrong kernel matrix: first column must be 0:sample_serial_number");
                }
                if ((int) prob.x[i][0].value <= 0 || (int) prob.x[i][0].value > max_index) {
                    throw new RuntimeException("Wrong input format: sample_serial_number out of range");
                }
            }

        fp.close();
        return prob;
    }

    public static SVMParameter buildParam() {
        SVMPrintInterface printFunc = svm_print_null;    // default printing to stdout

        SVMParameter param = new SVMParameter();
        // default values
        param.svm_type = SVMParameter.C_SVC;
        param.kernel_type = SVMParameter.RBF;
        param.degree = 3;
        param.gamma = 0;    // 1/num_features
        param.coef0 = 0;
        param.nu = 0.5;
        param.cache_size = 100;
        param.C = 1;
        param.eps = 1e-3;
        param.p = 0.1;
        param.shrinking = 1;
        param.probability = 0;
        param.nr_weight = 0;
        param.weight_label = new int[0];
        param.weight = new double[0];

        SVM.svm_set_print_string_function(printFunc);
        return param;
    }
}
