/*
 *	PsBusinessListener.java
 *
 *	All Rights Reserved, Copyright(c) FUJITSU FRONTECH LIMITED 2013
 */

package com.fujitsu.frontech.palmsecure_smpl.event;

import com.fujitsu.frontech.palmsecure_smpl.data.PsThreadResult;


public interface PsBusinessListener {

	void Ps_Sample_Apl_Java_NotifyWorkMessage(String messageKey);
	void Ps_Sample_Apl_Java_NotifyWorkMessage(String messageKey, int count);
	void Ps_Sample_Apl_Java_NotifyGuidance(String guidanceKey, boolean error);

	void Ps_Sample_Apl_Java_NotifyResult_Enroll(PsThreadResult stResult, int enrollscore);
	void Ps_Sample_Apl_Java_NotifyResult_Verify(PsThreadResult stResult);
	void Ps_Sample_Apl_Java_NotifyResult_Identify(PsThreadResult stResult);
	void Ps_Sample_Apl_Java_NotifyResult_Cancel(PsThreadResult stResult);
}
