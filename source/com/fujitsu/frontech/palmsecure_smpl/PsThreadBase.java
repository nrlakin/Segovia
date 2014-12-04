/*
 *	PsThreadBase.java
 *
 *	All Rights Reserved, Copyright(c) FUJITSU FRONTECH LIMITED 2013
 */

package com.fujitsu.frontech.palmsecure_smpl;

import com.fujitsu.frontech.palmsecure_smpl.data.PsThreadResult;
import com.fujitsu.frontech.palmsecure_smpl.event.PsBusinessListener;

import com.fujitsu.frontech.palmsecure.*;

public abstract class PsThreadBase extends Thread {

	protected PsBusinessListener businesslistener = null;
	protected PalmSecureIf palmsecureIf = null;
	protected JAVA_uint32 moduleHandle = null;
	protected String userID = null;
	protected PsMainFrame frame = null;

	protected static final String  NEW_LINE_CODE = System.getProperty("line.separator");

	protected PsThreadBase(PsMainFrame frame, PsBusinessListener businesslistener, PalmSecureIf palmsecureIf, JAVA_uint32 moduleHandle, String userID) {

		this.palmsecureIf = palmsecureIf;
		this.frame = frame;
		this.businesslistener = businesslistener;
		this.moduleHandle = moduleHandle;
		this.userID = userID;
	}

	protected void Ps_Sample_Apl_Java_NotifyWorkMessage(String processKey) {

		if (businesslistener != null) {
			businesslistener.Ps_Sample_Apl_Java_NotifyWorkMessage(processKey);
		}
	}

	protected void Ps_Sample_Apl_Java_NotifyWorkMessage(String processKey, int count) {

		if (businesslistener != null) {
			businesslistener.Ps_Sample_Apl_Java_NotifyWorkMessage(processKey, count);
		}
	}

	protected void Ps_Sample_Apl_Java_NotifyGuidance(String guidanceKey, boolean error) {

		if (businesslistener != null) {
			businesslistener.Ps_Sample_Apl_Java_NotifyGuidance(guidanceKey, false);
		}
	}

	protected void Ps_Sample_Apl_Java_NotifyResult_Enroll(PsThreadResult stResult, int enrollscore) {

		if (businesslistener != null) {
			businesslistener.Ps_Sample_Apl_Java_NotifyResult_Enroll(stResult, enrollscore);
		}
	}

	protected void Ps_Sample_Apl_Java_NotifyResult_Verify(PsThreadResult stResult) {

		if (businesslistener != null) {
			businesslistener.Ps_Sample_Apl_Java_NotifyResult_Verify(stResult);
		}
	}

	protected void Ps_Sample_Apl_Java_NotifyResult_Identify(PsThreadResult stResult) {

		if (businesslistener != null) {
			businesslistener.Ps_Sample_Apl_Java_NotifyResult_Identify(stResult);
		}
	}

	protected void Ps_Sample_Apl_Java_NotifyResult_Cancel(PsThreadResult stResult) {

		if (businesslistener != null) {
			businesslistener.Ps_Sample_Apl_Java_NotifyResult_Cancel(stResult);
		}
	}
}
