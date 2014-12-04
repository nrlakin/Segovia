/*
 * PsStreamingCallback.java
 *
 *	All Rights Reserved, Copyright(c) FUJITSU FRONTECH LIMITED 2013
 */

package com.fujitsu.frontech.palmsecure_smpl;

import com.fujitsu.frontech.palmsecure.*;
import com.fujitsu.frontech.palmsecure.util.*;

public class PsStreamingCallback implements JAVA_BioAPI_GUI_STREAMING_CALLBACK_IF{

	public long JAVA_BioAPI_GUI_STREAMING_CALLBACK(
			Object GuiStreamingCallbackCtx, JAVA_BioAPI_GUI_BITMAP Bitmap) {

		PsMainFrame frame = (PsMainFrame) GuiStreamingCallbackCtx;
		frame.Ps_Sample_Apl_Java_SetSilhouette(Bitmap);

		return PalmSecureConstant.JAVA_BioAPI_OK;
	}



}
