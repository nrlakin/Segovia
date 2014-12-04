/*
 *	PsThreadCancel.java
 *
 *	All Rights Reserved, Copyright(c) FUJITSU FRONTECH LIMITED 2013
 */

package com.fujitsu.frontech.palmsecure_smpl;

import com.fujitsu.frontech.palmsecure_smpl.data.PsThreadResult;
import com.fujitsu.frontech.palmsecure_smpl.event.PsBusinessListener;

import com.fujitsu.frontech.palmsecure.*;
import com.fujitsu.frontech.palmsecure.util.*;

public class PsThreadCancel extends PsThreadBase {

	public PsThreadCancel(PsMainFrame frame, PsBusinessListener businesslistener, PalmSecureIf palmsecureIf, JAVA_uint32 moduleHandle) {

		super(frame, businesslistener, palmsecureIf, moduleHandle, "");
	}

	public void run() {

		PsThreadResult stResult = new PsThreadResult();

		//Cancel
		///////////////////////////////////////////////////////////////////////////
		try {
			stResult.result = palmsecureIf.JAVA_PvAPI_Cancel(
					moduleHandle,
					stResult.errInfo);
			if (stResult.result != PalmSecureConstant.JAVA_BioAPI_OK) {
				return;
			}
		} catch(PalmSecureException e) {
		}
		///////////////////////////////////////////////////////////////////////////

		return;
	}
}
