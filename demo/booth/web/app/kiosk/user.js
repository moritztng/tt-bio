// Firefox preferences for the booth kiosk. launch.sh copies this into a fresh profile on every
// start. Each block names the dialog or surface it keeps off the screen.

// first run, default browser, what's new, welcome
user_pref("browser.shell.checkDefaultBrowser", false);
user_pref("browser.startup.homepage_override.mstone", "ignore");
user_pref("startup.homepage_welcome_url", "");
user_pref("startup.homepage_welcome_url.additional", "");
user_pref("browser.aboutwelcome.enabled", false);
user_pref("trailhead.firstrun.didSeeAboutWelcome", true);
user_pref("datareporting.policy.dataSubmissionEnabled", false);
user_pref("datareporting.healthreport.uploadEnabled", false);
user_pref("toolkit.telemetry.reportingpolicy.firstRun", false);

// crashes: no restore-session page, no safe-mode prompt, no crash-report bar
user_pref("browser.sessionstore.resume_from_crash", false);
user_pref("browser.sessionstore.max_resumed_crashes", -1);
user_pref("toolkit.startup.max_resumed_crashes", -1);
user_pref("browser.crashReports.unsubmittedCheck.enabled", false);
user_pref("browser.crashReports.unsubmittedCheck.autoSubmit2", false);
user_pref("browser.tabs.crashReporting.sendReport", false);

// updates (the snap itself is held by the ops side: snap refresh --hold firefox)
user_pref("app.update.auto", false);
user_pref("app.update.checkInstallTime", false);
user_pref("extensions.update.enabled", false);
user_pref("app.normandy.enabled", false);
user_pref("app.shield.optoutstudies.enabled", false);
user_pref("browser.region.update.enabled", false);

// quitting and closing: no shortcut, no prompt
user_pref("browser.quitShortcut.disabled", true);
user_pref("browser.warnOnQuit", false);
user_pref("browser.warnOnQuitShortcut", false);
user_pref("browser.tabs.warnOnClose", false);
user_pref("dom.disable_beforeunload", true);

// full screen with no toast and no transition
user_pref("full-screen-api.warning.timeout", 0);
user_pref("full-screen-api.warning.delay", -1);
user_pref("full-screen-api.transition-duration.enter", "0 0");
user_pref("full-screen-api.transition-duration.leave", "0 0");

// an empty window or a page still loading paints the app's ground, never white
user_pref("browser.startup.blankWindow", false);
user_pref("browser.display.background_color", "#08090C");
user_pref("browser.display.background_color.dark", "#08090C");
user_pref("browser.display.use_system_colors", false);
user_pref("ui.systemUsesDarkTheme", 1);
user_pref("layout.css.prefers-color-scheme.content-override", 0);
user_pref("browser.theme.content-theme", 0);
user_pref("browser.theme.toolbar-theme", 0);
user_pref("toolkit.legacyUserProfileCustomizations.stylesheets", true);   // userChrome.css

// zoom, pinch, swipe
user_pref("zoom.minPercent", 100);
user_pref("zoom.maxPercent", 100);
user_pref("apz.allow_zooming", false);
user_pref("browser.gesture.pinch.in", "");
user_pref("browser.gesture.pinch.out", "");
user_pref("browser.gesture.swipe.left", "");
user_pref("browser.gesture.swipe.right", "");
user_pref("dom.w3c_touch_events.enabled", 1);

// keys a visitor might press: menu bar on Alt, find-as-you-type, devtools
user_pref("ui.key.menuAccessKeyFocuses", false);
user_pref("accessibility.typeaheadfind", false);
user_pref("accessibility.typeaheadfind.manual", false);
user_pref("devtools.policy.disabled", true);

// popups and bars: notifications, translation, pocket, password and form prompts
user_pref("permissions.default.desktop-notification", 2);
user_pref("dom.webnotifications.enabled", false);
user_pref("browser.translations.enable", false);
user_pref("browser.translations.automaticallyPopup", false);
user_pref("extensions.pocket.enabled", false);
user_pref("signon.rememberSignons", false);
user_pref("browser.formfill.enable", false);
user_pref("browser.newtabpage.enabled", false);

// offline booth: no captive-portal or connectivity banner, no network lookups
user_pref("network.captive-portal-service.enabled", false);
user_pref("network.connectivity-service.enabled", false);
user_pref("browser.safebrowsing.malware.enabled", false);
user_pref("browser.safebrowsing.phishing.enabled", false);
user_pref("browser.safebrowsing.downloads.enabled", false);
user_pref("network.dns.disableIPv6", true);
user_pref("network.prefetch-next", false);

// GPU: WebGL on the iGPU through WebRender
user_pref("webgl.force-enabled", true);
user_pref("gfx.webrender.all", true);
user_pref("webgl.enable-debug-renderer-info", true);

// snap sandbox: no desktop portals (none run in a kiosk session)
user_pref("widget.use-xdg-desktop-portal.settings", 0);
user_pref("widget.use-xdg-desktop-portal.file-picker", 0);
user_pref("widget.use-xdg-desktop-portal.mime-handler", 0);
user_pref("widget.use-xdg-desktop-portal.open-uri", 0);
user_pref("widget.use-xdg-desktop-portal.location", 0);

// the app logs its self-test and warnings to the browser's stdout
user_pref("devtools.console.stdout.content", true);
