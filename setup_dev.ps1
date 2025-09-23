# setup-dev.ps1
# Installs Python, VS 2022 Build Tools (C++), SDL2, CUDA Toolkit, Node.js
# Supports configuration file for version management
# Works in Windows PowerShell 5.1 and PowerShell 7+.

[CmdletBinding()]
param(
  [switch]$SkipVS,
  [switch]$SkipCUDA,
  [switch]$SkipSDL,
  [switch]$SkipNode,           # Skip Node.js installation
  [switch]$SkipPython,         # Skip Python installation
  [switch]$ForceReinstall,      # Force reinstall all components
  [switch]$ForceVS,            # Force reinstall VS Build Tools
  [switch]$ForceCUDA,          # Force reinstall CUDA
  [switch]$ForceSDL,           # Force reinstall SDL2
  [switch]$ForceSDL2,          # Alias for ForceSDL (for batch script compatibility)
  [switch]$ForceNode,          # Force reinstall Node.js
  [switch]$ForcePython,        # Force reinstall Python
  [string]$ConfigFile = "setup-config.json",  # Configuration file path
  [string]$CudaVersion = "",    # Override config file version
  [string]$SdlVersion = "",     # Override config file version
  [string]$NodeVersion = "",    # Override config file version
  [string]$PythonVersion = "",  # Override config file version
  [string]$SdlRoot = ""         # Override config file path
)

# Handle ForceSDL2 alias
if ($ForceSDL2) { $ForceSDL = $true }

$ErrorActionPreference = 'Stop'

# Set up logging
$LogFile = Join-Path (Get-Location) 'build_packages.log'
function Write-Log {
    param([string]$Message)
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $logMessage = "[$timestamp] $Message"
    Write-Host $logMessage
    Add-Content -Path $LogFile -Value $logMessage -Encoding UTF8
}

# Initialize log file
"=== PianoidCore Development Environment Setup Log ===" | Set-Content -Path $LogFile -Encoding UTF8
Write-Log "Script started at $(Get-Date)"
Write-Log "Log file: $LogFile"

# Default versions (used when config file doesn't exist or values are missing)
$DefaultVersions = @{
  python = "3.12.0"
  cuda = "12.6.0"
  nodejs = "20.18.0"
  sdl2 = "2.30.8"
}

$DefaultPaths = @{
  sdl2_root = "C:\"
}

$DefaultCudaArchs = @("75", "80", "86", "89")

function Load-Configuration {
  param([string]$ConfigPath)
  
  $config = @{
    versions = $DefaultVersions.Clone()
    paths = $DefaultPaths.Clone()
    cuda = @{ architectures = $DefaultCudaArchs }
    options = @{
      skip_components = @()
      force_reinstall_components = @()
      auto_reboot = $false
      clean_install = $true
    }
  }
  
  if (Test-Path $ConfigPath) {
    Write-Log "Loading configuration from: $ConfigPath"
    try {
      $configData = Get-Content -Path $ConfigPath -Raw | ConvertFrom-Json
      
      # Merge versions (use config values, fall back to defaults)
      if ($configData.versions) {
        foreach ($key in $configData.versions.PSObject.Properties.Name) {
          if ($configData.versions.$key) {
            $config.versions[$key] = $configData.versions.$key
            Write-Log "Config: $key = $($configData.versions.$key)"
          }
        }
      }
      
      # Merge paths
      if ($configData.paths) {
        foreach ($key in $configData.paths.PSObject.Properties.Name) {
          if ($configData.paths.$key) {
            $config.paths[$key] = $configData.paths.$key
          }
        }
      }
      
      # Merge CUDA architectures
      if ($configData.cuda -and $configData.cuda.architectures) {
        $config.cuda.architectures = $configData.cuda.architectures
      }
      
      # Merge options
      if ($configData.options) {
        foreach ($key in $configData.options.PSObject.Properties.Name) {
          $config.options[$key] = $configData.options.$key
        }
      }
      
    } catch {
      Write-Log "Failed to parse config file: $_"
      Write-Log "Using default values"
    }
  } else {
    Write-Log "Config file not found: $ConfigPath"
    Write-Log "Using default versions"
  }
  
  # Apply command line overrides
  if ($PythonVersion) { $config.versions.python = $PythonVersion }
  if ($CudaVersion) { $config.versions.cuda = $CudaVersion }
  if ($NodeVersion) { $config.versions.nodejs = $NodeVersion }  
  if ($SdlVersion) { $config.versions.sdl2 = $SdlVersion }
  if ($SdlRoot) { $config.paths.sdl2_root = $SdlRoot }
  
  return $config
}

function Assert-Admin {
  $id = [Security.Principal.WindowsIdentity]::GetCurrent()
  $p  = [Security.Principal.WindowsPrincipal]$id
  if (-not $p.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Log "ERROR: This script requires Administrator privileges"
    throw "Please run this script in an elevated (Administrator) PowerShell."
  }
  Write-Log "Administrator privileges confirmed"
}

function Get-Tool($name) { 
  Get-Command $name -ErrorAction SilentlyContinue 
}

function Invoke-Download($Url, $OutFile) {
  Write-Log "Downloading $Url ..."
  try {
    Invoke-WebRequest -Uri $Url -OutFile $OutFile
    if (-not (Test-Path $OutFile)) { 
      throw "Download failed: $Url" 
    }
    Write-Log "Download completed: $OutFile"
  } catch {
    Write-Log "Download failed: $_"
    throw
  }
}

function Ensure-Python {
  param([string]$Version)
  
  if ($SkipPython) { 
    Write-Log "Skipping Python installation"
    return $null 
  }
  
  # Check if Python is already installed
  $forceRequested = $ForceReinstall -or $ForcePython
  $python = Get-Tool 'python'
  
  if ($python -and (-not $forceRequested)) {
    try {
      $currentVersion = & $python --version 2>&1
      if ($currentVersion -match "Python (\d+\.\d+\.\d+)") {
        $installedVersion = $Matches[1]
        Write-Log "Python already installed: $currentVersion"
        if ($installedVersion -eq $Version) {
          Write-Log "Correct version already installed."
          # Also check pip
          $pip = Get-Tool 'pip'
          if ($pip) {
            $pipVersion = & $pip --version
            Write-Log "pip available: $pipVersion"
          }
          return $python.Source
        } else {
          Write-Log "Different version found, will install Python $Version"
        }
      }
    } catch {
      Write-Log "Python found but version check failed, proceeding with installation"
    }
  }
  
  if ($forceRequested) {
    Write-Log "Force reinstall requested for Python"
  }

  Write-Log ""
  Write-Log "== Installing Python $Version =="
  
  # Try winget first
  $winget = Get-Tool 'winget'
  if ($winget) {
    try {
      Write-Log "Attempting to install Python $Version via winget..."
      $wingetOutput = & $winget install -e --id Python.Python.3.12 --version $Version --silent --accept-package-agreements --accept-source-agreements 2>&1
      $wingetOutput | ForEach-Object { Write-Log "winget: $_" }
      Start-Sleep -Seconds 5
      
      # Refresh PATH and test
      $env:PATH = [Environment]::GetEnvironmentVariable('PATH','Machine') + ';' + [Environment]::GetEnvironmentVariable('PATH','User')
      $python = Get-Tool 'python'
      
      if ($python) { 
        $installedVersion = & $python --version
        Write-Log "Python installed successfully via winget: $installedVersion"
        return $python.Source
      }
    } catch { 
      Write-Log "winget install failed, falling back to direct installer: $_"
    }
  }

  # Fallback to direct installer
  Write-Log "Using direct installer method..."
  
  # Determine architecture
  $arch = if ([Environment]::Is64BitOperatingSystem) { "amd64" } else { "win32" }
  Write-Log "Detected architecture: $arch"
  
  # Construct download URL
  $url = "https://www.python.org/ftp/python/$Version/python-$Version-$arch.exe"
  $tmp = Join-Path $env:TEMP "python-$Version-installer.exe"
  
  try {
    Write-Log "Downloading Python $Version..."
    Invoke-Download $url $tmp
    
    Write-Log "Running Python installer..."
    $installArgs = @(
      '/quiet',
      'InstallAllUsers=0',
      'PrependPath=1',
      'Include_test=0',
      'Include_pip=1',
      'Include_doc=1',
      'Include_dev=1',
      'Include_launcher=1'
    )
    
    $process = Start-Process $tmp -ArgumentList $installArgs -NoNewWindow -Wait -PassThru
    Write-Log "Python installer exit code: $($process.ExitCode)"
    
    # Cleanup installer
    Remove-Item $tmp -ErrorAction SilentlyContinue
    
    # Refresh PATH and test
    $env:PATH = [Environment]::GetEnvironmentVariable('PATH','Machine') + ';' + [Environment]::GetEnvironmentVariable('PATH','User')
    $python = Get-Tool 'python'
    
    if ($python) { 
      $installedVersion = & $python --version
      Write-Log "Python installed successfully: $installedVersion"
      
      # Verify pip
      $pip = Get-Tool 'pip'
      if ($pip) {
        $pipVersion = & $pip --version
        Write-Log "pip installed successfully: $pipVersion"
      }
      
      return $python.Source
    } else {
      throw "Python installation failed or not found in PATH"
    }
  } catch {
    Write-Log "Failed to install Python: $_"
    throw
  }
}

function Uninstall-CUDA {
  Write-Log ""
  Write-Log "== Uninstalling existing CUDA installations =="
  
  # Try winget uninstall first
  $winget = Get-Tool 'winget'
  if ($winget) {
    try {
      Write-Log "Attempting to uninstall CUDA via winget..."
      $wingetOutput = & $winget uninstall --id Nvidia.CUDA --silent --accept-source-agreements 2>&1
      $wingetOutput | ForEach-Object { Write-Log "winget: $_" }
    } catch {
      Write-Log "winget uninstall failed, continuing with manual cleanup: $_"
    }
  }

  # Manual cleanup of CUDA environment variables and common paths
  $cudaPaths = @(
    "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA",
    "C:\ProgramData\NVIDIA Corporation\CUDA Samples"
  )
  
  foreach ($path in $cudaPaths) {
    if (Test-Path $path) {
      Write-Log "Removing CUDA directory: $path"
      try {
        Remove-Item -Path $path -Recurse -Force -ErrorAction Continue
      } catch {
        Write-Log "Could not fully remove $path - some files may be in use: $_"
      }
    }
  }

  # Clear CUDA environment variables
  [Environment]::SetEnvironmentVariable('CUDA_PATH', $null, 'Machine')
  [Environment]::SetEnvironmentVariable('CUDA_PATH_V13_0', $null, 'Machine')
  [Environment]::SetEnvironmentVariable('CUDA_PATH_V12_6', $null, 'Machine')
  
  Write-Log "CUDA cleanup completed. A reboot may be required for full cleanup."
}

function Ensure-VSBuildTools {
  if ($SkipVS) { 
    Write-Log "Skipping Visual Studio Build Tools installation"
    return 
  }
  
  $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
  $needInstall = $true
  
  $forceRequested = $ForceReinstall -or $ForceVS
  
  if ((Test-Path $vswhere) -and (-not $forceRequested)) {
    $instPath = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if ($instPath) { 
      Write-Log "VS Build Tools found at: $instPath"
      $needInstall = $false 
    }
  }
  
  if ($forceRequested) {
    Write-Log "Force reinstall requested for VS Build Tools"
    $needInstall = $true
  }
  
  if ($needInstall) {
    Write-Log ""
    Write-Log "== Installing Visual Studio 2022 Build Tools (C++ workload) =="
    $tmp = Join-Path $env:TEMP "vs_buildtools.exe"
    Invoke-Download "https://aka.ms/vs/17/release/vs_buildtools.exe" $tmp
    $args = @(
      '--quiet','--wait','--norestart','--nocache',
      '--add','Microsoft.VisualStudio.Workload.VCTools',
      '--add','Microsoft.VisualStudio.Component.Windows10SDK.19041',
      '--add','Microsoft.VisualStudio.Component.VC.CMake.Project',
      '--add','Microsoft.VisualStudio.Component.VC.Tools.x86.x64',
      '--includeRecommended'
    )
    Write-Log "Running VS Build Tools installer with args: $($args -join ' ')"
    $process = Start-Process $tmp -ArgumentList $args -NoNewWindow -Wait -PassThru
    Write-Log "VS Build Tools installer exit code: $($process.ExitCode)"
  } else {
    Write-Log "VS Build Tools with C++ workload already present."
  }
}

function Get-VcBinHostx64x64 {
  Write-Log "Locating MSVC compiler binary path..."
  $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
  if (Test-Path $vswhere) {
    $inst = (& $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath).Trim()
    if ($inst) {
      $vcToolsRoot = Join-Path $inst 'VC\Tools\MSVC'
      $vcVer = (Get-ChildItem -Directory $vcToolsRoot | Sort-Object Name -Descending | Select-Object -First 1).FullName
      $bin = Join-Path $vcVer 'bin\Hostx64\x64'
      if (Test-Path $bin) { 
        Write-Log "Found MSVC bin path: $bin"
        return $bin 
      }
    }
  }
  
  $roots = @("$env:ProgramFiles\Microsoft Visual Studio", "${env:ProgramFiles(x86)}\Microsoft Visual Studio")
  foreach ($r in $roots) {
    if (Test-Path $r) {
      $cl = Get-ChildItem -Path $r -Recurse -Filter cl.exe -ErrorAction SilentlyContinue |
            Where-Object { $_.FullName -like '*\VC\Tools\MSVC\*\bin\Hostx64\x64\cl.exe' } |
            Select-Object -First 1
      if ($cl) { 
        Write-Log "Found MSVC bin path via search: $($cl.Directory.FullName)"
        return $cl.Directory.FullName 
      }
    }
  }
  Write-Log "ERROR: MSVC bin\Hostx64\x64 not found"
  throw "MSVC bin\Hostx64\x64 not found. Make sure the C++ workload is installed."
}

function Ensure-NodeJS {
  param([string]$Version)
  
  if ($SkipNode) { 
    Write-Log "Skipping Node.js installation"
    return $null 
  }
  
  # Check if Node.js is already installed
  $forceRequested = $ForceReinstall -or $ForceNode
  $node = Get-Tool 'node'
  
  if ($node -and (-not $forceRequested)) {
    $currentVersion = & $node --version
    Write-Log "Node.js already installed: $currentVersion"
    if ($currentVersion -like "v$Version*") {
      Write-Log "Correct version already installed."
      return $node.Source
    } else {
      Write-Log "Different version found, will install Node.js $Version"
    }
  }
  
  if ($forceRequested) {
    Write-Log "Force reinstall requested for Node.js"
  }

  Write-Log ""
  Write-Log "== Installing Node.js $Version =="
  $winget = Get-Tool 'winget'
  if ($winget) {
    try {
      Write-Log "Attempting to install Node.js $Version via winget..."
      $wingetOutput = & $winget install -e --id OpenJS.NodeJS --version $Version --silent --accept-package-agreements --accept-source-agreements 2>&1
      $wingetOutput | ForEach-Object { Write-Log "winget: $_" }
      Start-Sleep -Seconds 5
      
      # Refresh PATH and test
      $env:PATH = [Environment]::GetEnvironmentVariable('PATH','Machine') + ';' + [Environment]::GetEnvironmentVariable('PATH','User')
      $node = Get-Tool 'node'
      
      if ($node) { 
        $installedVersion = & $node --version
        Write-Log "Node.js installed successfully via winget: $installedVersion"
        return $node.Source
      }
    } catch { 
      Write-Log "winget install failed, falling back to direct installer: $_"
    }
  }

  # Fallback to direct installer
  Write-Log "Downloading Node.js $Version installer..."
  $tmp = Join-Path $env:TEMP "node-v$Version-x64.msi"
  $url = "https://nodejs.org/dist/v$Version/node-v$Version-x64.msi"
  
  try {
    Invoke-Download $url $tmp
    Write-Log "Running Node.js installer..."
    $process = Start-Process "msiexec.exe" -ArgumentList @('/i', $tmp, '/quiet', '/norestart') -NoNewWindow -Wait -PassThru
    Write-Log "Node.js installer exit code: $($process.ExitCode)"
    
    # Refresh PATH and test
    $env:PATH = [Environment]::GetEnvironmentVariable('PATH','Machine') + ';' + [Environment]::GetEnvironmentVariable('PATH','User')
    $node = Get-Tool 'node'
    
    if ($node) { 
      $installedVersion = & $node --version
      Write-Log "Node.js installed successfully: $installedVersion"
      return $node.Source
    } else {
      throw "Node.js installation failed or not found in PATH"
    }
  } catch {
    Write-Log "Failed to install Node.js: $_"
    throw
  }
}

function Ensure-SDL2 {
  param([string]$Version, [string]$SdlRoot)
  
  if ($SkipSDL) { 
    Write-Log "Skipping SDL2 installation"
    return $null 
  }
  
  $final = Join-Path $SdlRoot ("SDL2-$Version")
  $forceRequested = $ForceReinstall -or $ForceSDL
  
  if ($forceRequested) {
    Write-Log "Force reinstall requested for SDL2"
    if (Test-Path $final) {
      Write-Log "Removing existing SDL2 installation: $final"
      Remove-Item -Recurse -Force $final
    }
  }
  
  if ((Test-Path $final) -and (-not $forceRequested)) {
    Write-Log "SDL2 $Version already installed at: $final"
    return $final
  }
  
  Write-Log ""
  Write-Log "== Installing SDL2 $Version (VC) to $final =="

  $zip   = Join-Path $env:TEMP "SDL2-devel-$Version-VC.zip"
  $stage = Join-Path $env:TEMP ("sdl2_unpack_" + [guid]::NewGuid().ToString('N'))
  New-Item -ItemType Directory -Path $stage -Force | Out-Null

  Invoke-Download "https://www.libsdl.org/release/SDL2-devel-$Version-VC.zip" $zip
  Write-Log "Extracting SDL2 archive..."
  Expand-Archive -Path $zip -DestinationPath $stage -Force

  $src = (Get-ChildItem -Directory $stage | Where-Object { $_.Name -like 'SDL2-*' } |
          Sort-Object Name -Descending | Select-Object -First 1)
  if (-not $src) { 
    Write-Log "ERROR: SDL2 base folder not found after extraction"
    throw "SDL2 base folder not found after extraction." 
  }

  $parent = Split-Path -Path $final -Parent
  if ($parent -and (-not (Test-Path $parent))) {
    New-Item -ItemType Directory -Path $parent -Force | Out-Null
  }
  Write-Log "Moving SDL2 to final location: $final"
  Move-Item -Path $src.FullName -Destination $final

  # Cleanup staging
  Remove-Item -Recurse -Force $stage, $zip -ErrorAction SilentlyContinue
  Write-Log "SDL2 installation completed successfully"
  return $final
}

function Ensure-CUDA {
  param([string]$Version)
  
  if ($SkipCUDA) { 
    Write-Log "Skipping CUDA installation"
    return $env:CUDA_PATH 
  }
  
  # Check if force reinstall is requested
  $forceRequested = $ForceReinstall -or $ForceCUDA
  if ($forceRequested) {
    Write-Log "Force reinstall requested for CUDA"
    Uninstall-CUDA
    # Refresh environment variables after uninstall
    $env:CUDA_PATH = [Environment]::GetEnvironmentVariable('CUDA_PATH','Machine')
  }
  
  $cuda = [Environment]::GetEnvironmentVariable('CUDA_PATH','Machine')
  if (-not $cuda) { 
    $cuda = $env:CUDA_PATH 
  }
  
  if ($cuda -and (Test-Path $cuda) -and (-not $forceRequested)) { 
    Write-Log "CUDA already installed at: $cuda"
    return $cuda 
  }

  Write-Log ""
  Write-Log "== Installing CUDA Toolkit $Version =="
  $winget = Get-Tool 'winget'
  if ($winget) {
    try {
      Write-Log "Attempting to install CUDA $Version via winget..."
      $wingetOutput = & $winget install -e --id Nvidia.CUDA --version $Version --silent --accept-package-agreements --accept-source-agreements 2>&1
      $wingetOutput | ForEach-Object { Write-Log "winget: $_" }
      Start-Sleep -Seconds 10
      
      # Refresh environment variables
      $env:CUDA_PATH = [Environment]::GetEnvironmentVariable('CUDA_PATH','Machine')
      $cuda = $env:CUDA_PATH
      
      if ($cuda -and (Test-Path $cuda)) { 
        Write-Log "CUDA installed successfully via winget at: $cuda"
        return $cuda 
      }
    } catch { 
      Write-Log "winget install failed, falling back to direct installer: $_"
    }
  }

  # Fallback to direct installer
  Write-Log "Downloading CUDA $Version installer..."
  $tmp = Join-Path $env:TEMP "cuda_${Version}_windows_network.exe"
  $url = "https://developer.download.nvidia.com/compute/cuda/$Version/network_installers/cuda_${Version}_windows_network.exe"
  
  try {
    Invoke-Download $url $tmp
  } catch {
    # Try local installer as fallback
    $localUrl = "https://developer.download.nvidia.com/compute/cuda/$Version/local_installers/cuda_${Version}_windows.exe"
    $tmp = Join-Path $env:TEMP "cuda_${Version}_windows.exe"
    Write-Log "Network installer failed, trying local installer..."
    Invoke-Download $localUrl $tmp
  }
  
  Write-Log "Running CUDA installer (this may take several minutes)..."
  $process = Start-Process $tmp -ArgumentList @('-s','-n') -NoNewWindow -Wait -PassThru
  Write-Log "CUDA installer exit code: $($process.ExitCode)"

  # Refresh and find CUDA installation
  $env:CUDA_PATH = [Environment]::GetEnvironmentVariable('CUDA_PATH','Machine')
  $cuda = $env:CUDA_PATH
  
  if (-not ($cuda -and (Test-Path $cuda))) {
    $guess = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
    if (Test-Path $guess) {
      $cuda = (Get-ChildItem -Directory $guess | Where-Object { $_.Name -like 'v*' } |
               Sort-Object Name -Descending | Select-Object -First 1).FullName
      Write-Log "Found CUDA installation at: $cuda"
    }
  }
  
  if (-not ($cuda -and (Test-Path $cuda))) { 
    Write-Log "ERROR: CUDA Toolkit not found after install"
    throw "CUDA Toolkit not found after install; please verify and rerun." 
  }
  Write-Log "CUDA installation completed successfully"
  return $cuda
}

function Write-BuildConfig($CudaHome, $VcBin, $SdlBase, $CudaArchs) {
  $cfg = @{
    windows = @{
      cuda_home = $CudaHome
      visual_studio = @{ vc_tools_bin_hostx64_x64 = $VcBin }
      sdl2 = @{ base_path = $SdlBase }
    }
    cuda_arch_list = $CudaArchs
  }
  $out = Join-Path (Get-Location) 'build_config.json'
  $cfg | ConvertTo-Json -Depth 6 | Set-Content -Path $out -Encoding UTF8
  Write-Log ""
  Write-Log "build_config.json written to: $out"
  Write-Log "  cuda_home  : $CudaHome"
  Write-Log "  VC bin     : $VcBin"
  Write-Log "  SDL2 base  : $SdlBase"
}

# --- main ---
Write-Log "=== PianoidCore Development Environment Setup ==="

# Load configuration
$config = Load-Configuration $ConfigFile

Write-Log "Python Version: $($config.versions.python)"
Write-Log "CUDA Version: $($config.versions.cuda)"
Write-Log "Node.js Version: $($config.versions.nodejs)"
Write-Log "SDL2 Version: $($config.versions.sdl2)"

if ($ForceReinstall) { 
  Write-Log "Force reinstall: ALL components" 
}
elseif ($ForceVS -or $ForceCUDA -or $ForceSDL -or $ForceNode -or $ForcePython) {
  $components = @()
  if ($ForceVS) { $components += 'VS' }
  if ($ForceCUDA) { $components += 'CUDA' }
  if ($ForceSDL) { $components += 'SDL2' }
  if ($ForceNode) { $components += 'Node.js' }
  if ($ForcePython) { $components += 'Python' }
  Write-Log "Force reinstall: $($components -join ', ')"
}

try {
  Assert-Admin

  # Handle single component installs (skip others)
  if ($ForcePython -and -not $ForceReinstall) {
    $SkipVS = $true; $SkipCUDA = $true; $SkipSDL = $true; $SkipNode = $true
    Write-Log "Single component install: Python only"
  }
  elseif ($ForceCUDA -and -not $ForceReinstall) {
    $SkipVS = $true; $SkipPython = $true; $SkipSDL = $true; $SkipNode = $true
    Write-Log "Single component install: CUDA only"
  }
  elseif ($ForceNode -and -not $ForceReinstall) {
    $SkipVS = $true; $SkipCUDA = $true; $SkipSDL = $true; $SkipPython = $true
    Write-Log "Single component install: Node.js only"
  }
  elseif ($ForceSDL -and -not $ForceReinstall) {
    $SkipVS = $true; $SkipCUDA = $true; $SkipPython = $true; $SkipNode = $true
    Write-Log "Single component install: SDL2 only"
  }

  # Install in logical order
  Write-Log ""
  Write-Log "Starting component installation..."
  
  $python   = Ensure-Python -Version $config.versions.python

  if (-not $SkipVS) {
    Ensure-VSBuildTools
    $vcBin = Get-VcBinHostx64x64
    Write-Log "Testing MSVC compiler..."
    $clOutput = & (Join-Path $vcBin 'cl.exe') /? 2>&1
    Write-Log "MSVC compiler test successful"
  } else {
    $vcBin = $null
  }

  $sdlBase  = Ensure-SDL2 -Version $config.versions.sdl2 -SdlRoot $config.paths.sdl2_root
  $nodeJs   = Ensure-NodeJS -Version $config.versions.nodejs
  $cudaHome = Ensure-CUDA -Version $config.versions.cuda

  # Only write build config if we have all components
  if ($vcBin -and $cudaHome -and $sdlBase) {
    Write-BuildConfig -CudaHome $cudaHome -VcBin $vcBin -SdlBase $sdlBase -CudaArchs $config.cuda.architectures
  } else {
    Write-Log "Skipping build_config.json creation (some components were skipped)"
  }

  Write-Log ""
  Write-Log "=== Setup Complete ==="
  if ($python) { Write-Log "Python $($config.versions.python) installed at: $python" }
  if ($cudaHome) { Write-Log "CUDA $($config.versions.cuda) installed at: $cudaHome" }
  if ($nodeJs) { Write-Log "Node.js installed at: $nodeJs" }
  if ($sdlBase) { Write-Log "SDL2 installed at: $sdlBase" }
  Write-Log "Open a NEW terminal so PATH updates are visible, then build with:"
  Write-Log "  pip install -v ."
  Write-Log ""
  Write-Log "Script completed successfully at $(Get-Date)"

} catch {
  Write-Log "ERROR: Script failed with error: $_"
  Write-Log "Stack trace:"
  Write-Log $_.ScriptStackTrace
  Write-Log "Script failed at $(Get-Date)"
  throw
}