/*
 * PsSampleApl.java
 *
 *	All Rights Reserved, Copyright(c) FUJITSU FRONTECH LIMITED 2013
 */

package com.fujitsu.frontech.palmsecure_smpl;


public class PsSampleApl {

	public static void main(String[] args) {

		PsMainFrame mainFrame = new PsMainFrame();

		boolean result = mainFrame.Ps_Sample_Apl_Java();
		if (result != true) {
			System.exit(0);
		}
		mainFrame.setVisible(true);
	}
}
