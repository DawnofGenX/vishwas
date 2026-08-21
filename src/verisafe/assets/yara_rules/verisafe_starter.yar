/*
 * VeriSafe starter YARA ruleset — original rules written for VeriSafe.
 * License: MIT (same as the VeriSafe project).
 *
 * Scope: cheap, high-precision static patterns for the malware_file
 * capability's default bundle. Deliberately conservative — these flag
 * well-known test strings and generic suspicious constructs, NOT families
 * we cannot verify. Operators can extend via VERISAFE_YARA_RULES pointing
 * at a richer private bundle; this file is always loaded as the fallback.
 */

rule Eicar_Test_File
{
    meta:
        family = "eicar"
        author = "verisafe"
        description = "EICAR standard antivirus test file"
        severity = "test"
    strings:
        $eicar = "X5O!P%@AP[4\\PZX54(P^)7CC)7}$EICAR-STANDARD-ANTIVIRUS-TEST-FILE!$H+H*"
    condition:
        $eicar
}

rule Suspicious_AutoOpen_Dropper_Chain
{
    meta:
        family = "generic-dropper"
        author = "verisafe"
        description = "Script auto-execution hook combined with network fetch or shell spawn"
        severity = "low"
    strings:
        $autoexec_vbs = "AutoOpen" ascii nocase
        $autoexec_js = "Auto_Open" ascii nocase
        $shell = "WScript.Shell" ascii nocase
        $fetch1 = "URLDownloadToFileA" ascii nocase
        $fetch2 = "XmlHttp" ascii nocase wide
        $fetch3 = "requests.get(" ascii
    condition:
        1 of ($autoexec*) and $shell and 1 of ($fetch*)
}

rule PE_Suspicious_Imports_Combo
{
    meta:
        family = "generic-pe"
        author = "verisafe"
        description = "PE combining process injection + anti-debug API names in plaintext"
        severity = "low"
    strings:
        $inj1 = "VirtualAllocEx" ascii
        $inj2 = "WriteProcessMemory" ascii
        $inj3 = "CreateRemoteThread" ascii
        $dbg1 = "IsDebuggerPresent" ascii
        $dbg2 = "CheckRemoteDebuggerPresent" ascii
    condition:
        uint16(0) == 0x5A4D and 2 of ($inj*) and 1 of ($dbg*)
}
