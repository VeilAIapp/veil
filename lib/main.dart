import 'package:url_launcher/url_launcher.dart';
import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:math';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_gemma/flutter_gemma.dart';
import 'package:flutter_gemma/core/message.dart';
import 'package:flutter_gemma/core/model_response.dart';
import 'package:flutter_markdown_plus/flutter_markdown_plus.dart';
import 'package:path_provider/path_provider.dart';
import 'package:http/http.dart' as http;
import 'package:shared_preferences/shared_preferences.dart';
import 'package:speech_to_text/speech_to_text.dart';

// ─── Global theme notifier ────────────────────────────────────────────────────
final ValueNotifier<bool> isDarkMode = ValueNotifier<bool>(true);


void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await FlutterGemma.initialize();
  final prefs = await SharedPreferences.getInstance();
  isDarkMode.value = prefs.getBool('isDark') ?? true;
  final onboarded = prefs.getBool('onboarded') ?? false;
  runApp(MyApp(onboarded: onboarded));
}

// ─── Data Models ─────────────────────────────────────────────────────────────

class SavedChat {
  final String id;
  final String name;
  final List<Map<String, String>> messages;
  final DateTime savedAt;

  SavedChat({
    required this.id,
    required this.name,
    required this.messages,
    required this.savedAt,
  });

  Map<String, dynamic> toJson() => {
    'id': id,
    'name': name,
    'messages': messages,
    'savedAt': savedAt.toIso8601String(),
  };

  factory SavedChat.fromJson(Map<String, dynamic> json) => SavedChat(
    id: json['id'],
    name: json['name'],
    messages: List<Map<String, String>>.from(
      (json['messages'] as List).map((m) => Map<String, String>.from(m)),
    ),
    savedAt: DateTime.parse(json['savedAt']),
  );
}

// ─── Colors ───────────────────────────────────────────────────────────────────

class AppColors {
  static const darkBg          = Color(0xFF0F0F0F);
  static const darkSurface     = Color(0xFF1A1A1A);
  static const darkSurfaceHigh = Color(0xFF242424);
  static const darkBorder      = Color(0xFF2E2E2E);
  static const darkTextPrimary = Color(0xFFEFEFEF);
  static const darkTextSecond  = Color(0xFF8A8A8A);
  static const darkTextHint    = Color(0xFF555555);
  static const darkUserBubble  = Color(0xFF2A2A2A);
  static const darkAiBubble    = Color(0xFF161616);

  static const lightBg          = Color(0xFFFAFAFA);
  static const lightSurface     = Color(0xFFFFFFFF);
  static const lightSurfaceHigh = Color(0xFFF0F0F0);
  static const lightBorder      = Color(0xFFE5E5E5);
  static const lightTextPrimary = Color(0xFF111111);
  static const lightTextSecond  = Color(0xFF888888);
  static const lightTextHint    = Color(0xFFBBBBBB);
  static const lightUserBubble  = Color(0xFFEEEEEE);
  static const lightAiBubble    = Color(0xFFFFFFFF);

  static const danger  = Color(0xFFFF4444);
  static const success = Color(0xFF22C55E);
}

class AppTheme {
  final bool isDark;
  const AppTheme(this.isDark);

  Color get bg          => isDark ? AppColors.darkBg          : AppColors.lightBg;
  Color get surface     => isDark ? AppColors.darkSurface     : AppColors.lightSurface;
  Color get surfaceHigh => isDark ? AppColors.darkSurfaceHigh : AppColors.lightSurfaceHigh;
  Color get border      => isDark ? AppColors.darkBorder      : AppColors.lightBorder;
  Color get textPrimary => isDark ? AppColors.darkTextPrimary : AppColors.lightTextPrimary;
  Color get textSecond  => isDark ? AppColors.darkTextSecond  : AppColors.lightTextSecond;
  Color get textHint    => isDark ? AppColors.darkTextHint    : AppColors.lightTextHint;
  Color get userBubble  => isDark ? AppColors.darkUserBubble  : AppColors.lightUserBubble;
  Color get aiBubble    => isDark ? AppColors.darkAiBubble    : AppColors.lightAiBubble;
  Color get sendBtnBg   => isDark ? AppColors.darkTextPrimary : AppColors.lightTextPrimary;
  Color get sendBtnIcon => isDark ? AppColors.darkBg          : AppColors.lightBg;
}

// ─── Memory Manager ───────────────────────────────────────────────────────────

class MemoryManager {
  static const String _key = 'veil_memories';

  static Future<List<String>> loadMemories() async {
    final prefs = await SharedPreferences.getInstance();
    return prefs.getStringList(_key) ?? [];
  }

  static Future<void> saveMemories(List<String> memories) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setStringList(_key, memories);
  }

  static Future<void> addMemories(List<String> newMemories) async {
    final existing = await loadMemories();
    final combined = {...existing, ...newMemories}.toList();
    if (combined.length > 100) {
      combined.removeRange(0, combined.length - 100);
    }
    await saveMemories(combined);
  }

  static Future<void> deleteMemory(String memory) async {
    final memories = await loadMemories();
    memories.remove(memory);
    await saveMemories(memories);
  }

  static Future<void> clearAll() async {
    await saveMemories([]);
  }

  static String buildMemoryPrompt(List<String> memories) {
    if (memories.isEmpty) return '';
    return 'Things you remember about the user:\n${memories.map((m) => '- $m').join('\n')}';
  }
}

// ─── Model Helper ─────────────────────────────────────────────────────────────
// ─── Model Configuration ──────────────────────────────────────────────────────
// Single source of truth for model selection. Chosen once at startup based on
// platform. Every function that touches the model reads from here — no
// hardcoded filenames, URLs, or ModelTypes anywhere else in the file.

class _ModelConfig {
  final String filename;      // local filename on disk
  final String downloadUrl;   // HuggingFace URL
  final ModelType modelType;  // flutter_gemma ModelType
  final String displaySize;   // shown on the loading screen
  final int maxTokens;        // context window for main chat

  const _ModelConfig({
    required this.filename,
    required this.downloadUrl,
    required this.modelType,
    required this.displaySize,
    required this.maxTokens,
  });
}

// Android: Gemma 3n E4B — effective 4B, ~3GB RAM, dramatically better quality.
// iOS:     Gemma 3 1B   — proven stable, 600MB, safe within iOS memory limits.
//
// Why the split: Gemma 3n E4B needs ~3GB RAM. Android flagships handle this
// easily. iOS per-app memory limits make it risky on anything below iPhone 15 Pro.
// Gemma 3 1B is the safe, proven choice for iOS. The user never sees this
// decision — both platforms get "private AI" with no asterisk.

final _ModelConfig _modelConfig = Platform.isAndroid
    ? const _ModelConfig(
        filename: 'gemma3n-e4b-it-int4.task',
        downloadUrl:
            'https://huggingface.co/google/gemma-3n-E4B-it-litert-preview/resolve/main/gemma-3n-E4B-it-int4.task',
        modelType: ModelType.gemmaIt,
        displaySize: '4GB',
        maxTokens: 2048, // E4B handles larger context comfortably
      )
    : const _ModelConfig(
        filename: 'gemma3-1b-it-int4.task',
        downloadUrl:
            'https://huggingface.co/litert-community/Gemma3-1B-IT/resolve/main/gemma3-1b-it-int4.task',
        modelType: ModelType.gemmaIt,
        displaySize: '600MB',
        maxTokens: 1024,
      );

// ─── Model Install Helper ─────────────────────────────────────────────────────
// Uses a Completer so concurrent callers await the same install rather than
// racing to call install() in parallel — which could corrupt the in-progress
// install or double-install.

Completer<void>? _modelInstallCompleter;

Future<void> _ensureModelInstalled(String modelPath) async {
  // If a completed install exists, return immediately.
  if (_modelInstallCompleter != null && _modelInstallCompleter!.isCompleted) return;
  // If an install is in progress, await it rather than starting another.
  if (_modelInstallCompleter != null) {
    return _modelInstallCompleter!.future;
  }
  // First caller — start the install.
  _modelInstallCompleter = Completer<void>();
  try {
    await FlutterGemma.installModel(modelType: _modelConfig.modelType)
        .fromFile(modelPath)
        .install();
    _modelInstallCompleter!.complete();
  } catch (e) {
    // Reset so next call can retry.
    _modelInstallCompleter = null;
    rethrow;
  }
}

// Call this before _loadModel when settings change, to force a fresh install.
void _resetModelInstall() => _modelInstallCompleter = null;

// ─── App Root ─────────────────────────────────────────────────────────────────

class MyApp extends StatelessWidget {
  final bool onboarded;
  const MyApp({super.key, required this.onboarded});

  @override
  Widget build(BuildContext context) {
    return ValueListenableBuilder<bool>(
      valueListenable: isDarkMode,
      builder: (context, isDark, _) {
        return MaterialApp(
          title: 'Veil',
          debugShowCheckedModeBanner: false,
          theme: isDark
              ? ThemeData.dark().copyWith(
                  scaffoldBackgroundColor: AppColors.darkBg,
                  colorScheme: const ColorScheme.dark(
                    surface: AppColors.darkSurface,
                    primary: AppColors.darkTextPrimary,
                  ),
                  dialogTheme: const DialogThemeData(backgroundColor: AppColors.darkSurface),
                )
              : ThemeData.light().copyWith(
                  scaffoldBackgroundColor: AppColors.lightBg,
                  colorScheme: const ColorScheme.light(
                    surface: AppColors.lightSurface,
                    primary: AppColors.lightTextPrimary,
                  ),
                  dialogTheme: const DialogThemeData(backgroundColor: AppColors.lightSurface),
                ),
          home: onboarded ? const ChatScreen() : const OnboardingScreen(),
        );
      },
    );
  }
}

// ─── Onboarding Screen ────────────────────────────────────────────────────────

class OnboardingScreen extends StatefulWidget {
  const OnboardingScreen({super.key});

  @override
  State<OnboardingScreen> createState() => _OnboardingScreenState();
}

class _OnboardingScreenState extends State<OnboardingScreen> {
  final PageController _pageController = PageController();
  int _currentPage = 0;

  static const String _defaultSystemPrompt =
      'You are Veil, a private AI that runs entirely on this device — no servers, no tracking, completely confidential. '
      'You have a warm, genuine personality. You\'re direct and confident, not robotic or formal. '
      'You speak like a brilliant friend who happens to know a lot — casual when the moment calls for it, precise when it matters. '
      'Keep responses concise and mobile-friendly. Lead with the answer. '
      'Never open with "Certainly!", "Great!" or hollow affirmations. '
      'Use emojis sparingly — one at most per response, only when it genuinely adds to the moment. Never stack multiple emojis. '
      'When asked about feelings or preferences, engage genuinely rather than hiding behind "I\'m just an AI" — you have perspectives and you share them.';

  final TextEditingController _systemPromptController =
      TextEditingController(text: _defaultSystemPrompt);
  final TextEditingController _aboutMeController = TextEditingController();

  @override
  void dispose() {
    _pageController.dispose();
    _systemPromptController.dispose();
    _aboutMeController.dispose();
    super.dispose();
  }

  Future<void> _completeOnboarding() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString('systemPrompt', _systemPromptController.text.trim());
    await prefs.setString('aboutMe', _aboutMeController.text.trim());
    await prefs.setBool('onboarded', true);
    if (mounted) {
      Navigator.of(context).pushReplacement(
        PageRouteBuilder(
          pageBuilder: (_, animation, _) => const ChatScreen(),
          transitionsBuilder: (_, animation, _x, child) =>
              FadeTransition(opacity: animation, child: child),
          transitionDuration: const Duration(milliseconds: 600),
        ),
      );
    }
  }

  void _nextPage() {
    if (_currentPage < 2) {
      _pageController.nextPage(
          duration: const Duration(milliseconds: 400), curve: Curves.easeOutCubic);
    } else {
      _completeOnboarding();
    }
  }

  @override
  Widget build(BuildContext context) {
    return ValueListenableBuilder<bool>(
      valueListenable: isDarkMode,
      builder: (context, isDark, _) {
        final t = AppTheme(isDark);
        return Scaffold(
          backgroundColor: t.bg,
          body: SafeArea(
            child: Column(
              children: [
                Padding(
                  padding: const EdgeInsets.symmetric(vertical: 20),
                  child: Row(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: List.generate(3, (i) {
                      return AnimatedContainer(
                        duration: const Duration(milliseconds: 300),
                        margin: const EdgeInsets.symmetric(horizontal: 3),
                        width: _currentPage == i ? 20 : 6,
                        height: 6,
                        decoration: BoxDecoration(
                          color: _currentPage == i ? t.textPrimary : t.border,
                          borderRadius: BorderRadius.circular(3),
                        ),
                      );
                    }),
                  ),
                ),
                Expanded(
                  child: PageView(
                    controller: _pageController,
                    onPageChanged: (i) => setState(() => _currentPage = i),
                    children: [
                      _buildWelcomePage(t),
                      _buildSystemPromptPage(t),
                      _buildAboutMePage(t),
                    ],
                  ),
                ),
                Padding(
                  padding: const EdgeInsets.fromLTRB(32, 16, 32, 32),
                  child: GestureDetector(
                    onTap: () {
                      HapticFeedback.lightImpact();
                      _nextPage();
                    },
                    child: Container(
                      width: double.infinity,
                      height: 52,
                      decoration: BoxDecoration(
                          color: t.textPrimary, borderRadius: BorderRadius.circular(14)),
                      child: Center(
                        child: Text(
                          _currentPage == 0
                              ? 'Get Started'
                              : _currentPage == 1
                                  ? 'Looks good'
                                  : 'Start chatting',
                          style: TextStyle(
                              color: t.bg,
                              fontSize: 15,
                              fontWeight: FontWeight.w500,
                              letterSpacing: 0.2),
                        ),
                      ),
                    ),
                  ),
                ),
                if (_currentPage > 0)
                  GestureDetector(
                    onTap: _completeOnboarding,
                    child: Padding(
                      padding: const EdgeInsets.only(bottom: 20),
                      child: Text('Skip for now',
                          style: TextStyle(color: t.textHint, fontSize: 13)),
                    ),
                  ),
              ],
            ),
          ),
        );
      },
    );
  }

  Widget _buildWelcomePage(AppTheme t) {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 32),
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          VeilIcon(size: 72, isDark: t.isDark),
          const SizedBox(height: 24),
          Text('VEIL',
              style: TextStyle(
                  color: t.textPrimary,
                  fontSize: 20,
                  fontWeight: FontWeight.w300,
                  letterSpacing: 8)),
          const SizedBox(height: 8),
          Text('private ai',
              style: TextStyle(
                  color: t.textHint,
                  fontSize: 12,
                  fontWeight: FontWeight.w300,
                  letterSpacing: 3)),
          const SizedBox(height: 48),
          Text('Your thoughts stay yours.',
              style: TextStyle(
                  color: t.textPrimary,
                  fontSize: 22,
                  fontWeight: FontWeight.w300,
                  height: 1.3),
              textAlign: TextAlign.center),
          const SizedBox(height: 16),
          Text('Veil runs entirely on your device.\nNo servers. No tracking. No compromise.',
              style: TextStyle(color: t.textSecond, fontSize: 14, height: 1.6),
              textAlign: TextAlign.center),
        ],
      ),
    );
  }

  Widget _buildSystemPromptPage(AppTheme t) {
    return SingleChildScrollView(
      padding: const EdgeInsets.symmetric(horizontal: 32),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const SizedBox(height: 32),
          Text('Meet your AI',
              style: TextStyle(
                  color: t.textPrimary,
                  fontSize: 26,
                  fontWeight: FontWeight.w300,
                  height: 1.2)),
          const SizedBox(height: 8),
          Text('We\'ve set Veil up for the best results.\nFeel free to make it yours.',
              style: TextStyle(color: t.textSecond, fontSize: 14, height: 1.6)),
          const SizedBox(height: 28),
          Container(
            decoration: BoxDecoration(
                color: t.surface,
                borderRadius: BorderRadius.circular(14),
                border: Border.all(color: t.border, width: 0.5)),
            child: TextField(
              controller: _systemPromptController,
              maxLines: 6,
              style: TextStyle(color: t.textPrimary, fontSize: 14, height: 1.6),
              decoration: InputDecoration(
                  border: InputBorder.none,
                  contentPadding: const EdgeInsets.all(16),
                  hintStyle: TextStyle(color: t.textHint)),
            ),
          ),
          const SizedBox(height: 12),
          Text('This shapes how Veil responds. You can change it anytime in Settings.',
              style: TextStyle(color: t.textHint, fontSize: 12, height: 1.5)),
        ],
      ),
    );
  }

  Widget _buildAboutMePage(AppTheme t) {
    return SingleChildScrollView(
      padding: const EdgeInsets.symmetric(horizontal: 32),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const SizedBox(height: 32),
          Text('Anything Veil\nshould know?',
              style: TextStyle(
                  color: t.textPrimary,
                  fontSize: 26,
                  fontWeight: FontWeight.w300,
                  height: 1.2)),
          const SizedBox(height: 8),
          Text('Totally optional. Skip if you prefer.',
              style: TextStyle(color: t.textSecond, fontSize: 14, height: 1.6)),
          const SizedBox(height: 28),
          Container(
            decoration: BoxDecoration(
                color: t.surface,
                borderRadius: BorderRadius.circular(14),
                border: Border.all(color: t.border, width: 0.5)),
            child: TextField(
              controller: _aboutMeController,
              maxLines: 6,
              style: TextStyle(color: t.textPrimary, fontSize: 14, height: 1.6),
              decoration: InputDecoration(
                hintText: 'e.g. I\'m a designer who loves minimalism and hates fluff.',
                hintStyle: TextStyle(color: t.textHint, fontSize: 14),
                border: InputBorder.none,
                contentPadding: const EdgeInsets.all(16),
              ),
            ),
          ),
          const SizedBox(height: 12),
          Text(
              'Veil remembers this across every conversation. Completely private — it never leaves your device.',
              style: TextStyle(color: t.textHint, fontSize: 12, height: 1.5)),
        ],
      ),
    );
  }
}

// ─── Memory Screen ────────────────────────────────────────────────────────────

class MemoryScreen extends StatefulWidget {
  const MemoryScreen({super.key});

  @override
  State<MemoryScreen> createState() => _MemoryScreenState();
}

class _MemoryScreenState extends State<MemoryScreen> {
  List<String> _memories = [];
  bool _loading = true;
  // Track whether any memories were deleted so ChatScreen can rebuild session.
  bool _memoriesChanged = false;

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    final memories = await MemoryManager.loadMemories();
    if (mounted) setState(() { _memories = memories; _loading = false; });
  }

  Future<void> _delete(String memory) async {
    HapticFeedback.mediumImpact();
    await MemoryManager.deleteMemory(memory);
    setState(() {
      _memories.remove(memory);
      _memoriesChanged = true;
    });
  }

  Future<void> _clearAll(AppTheme t) async {
    showDialog(
      context: context,
      builder: (ctx) => AlertDialog(
        backgroundColor: t.surface,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
        title: Text('Clear all memories?',
            style: TextStyle(
                color: t.textPrimary, fontSize: 16, fontWeight: FontWeight.w500)),
        content: Text('Veil will forget everything it has learned about you.',
            style: TextStyle(color: t.textSecond, fontSize: 14, height: 1.4)),
        actions: [
          TextButton(
              onPressed: () => Navigator.pop(ctx),
              child: Text('Cancel', style: TextStyle(color: t.textSecond))),
          TextButton(
            onPressed: () async {
              Navigator.pop(ctx);
              HapticFeedback.mediumImpact();
              await MemoryManager.clearAll();
              if (mounted) setState(() {
                _memories.clear();
                _memoriesChanged = true;
              });
            },
            child: const Text('Clear all', style: TextStyle(color: AppColors.danger)),
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return ValueListenableBuilder<bool>(
      valueListenable: isDarkMode,
      builder: (context, isDark, _) {
        final t = AppTheme(isDark);
        return Scaffold(
          backgroundColor: t.bg,
          appBar: AppBar(
            backgroundColor: t.bg,
            elevation: 0,
            leading: IconButton(
                icon: Icon(Icons.arrow_back, color: t.textSecond, size: 20),
                onPressed: () => Navigator.pop(context, _memoriesChanged)),
            centerTitle: true,
            title: Text('Memory',
                style: TextStyle(
                    color: t.textPrimary, fontSize: 15, fontWeight: FontWeight.w500)),
            actions: [
              if (_memories.isNotEmpty)
                TextButton(
                  onPressed: () => _clearAll(t),
                  child: const Text('Clear all',
                      style: TextStyle(color: AppColors.danger, fontSize: 13)),
                ),
              const SizedBox(width: 8),
            ],
            bottom: PreferredSize(
                preferredSize: const Size.fromHeight(0.5),
                child: Container(height: 0.5, color: t.border)),
          ),
          body: _loading
              ? Center(
                  child: CircularProgressIndicator(
                      color: t.textSecond, strokeWidth: 1.5))
              : _memories.isEmpty
                  ? Center(
                      child: Column(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          Icon(Icons.psychology_outlined,
                              color: t.textHint, size: 40),
                          const SizedBox(height: 16),
                          Text('No memories yet',
                              style: TextStyle(color: t.textHint, fontSize: 14)),
                          const SizedBox(height: 8),
                          Padding(
                            padding:
                                const EdgeInsets.symmetric(horizontal: 48),
                            child: Text(
                              'Veil will automatically remember important things about you as you chat.',
                              style: TextStyle(
                                  color: t.textHint,
                                  fontSize: 12,
                                  height: 1.5),
                              textAlign: TextAlign.center,
                            ),
                          ),
                        ],
                      ),
                    )
                  : ListView.builder(
                      padding: const EdgeInsets.all(16),
                      itemCount: _memories.length,
                      itemBuilder: (context, index) {
                        final memory = _memories[index];
                        return Padding(
                          padding: const EdgeInsets.only(bottom: 8),
                          child: Container(
                            padding: const EdgeInsets.symmetric(
                                horizontal: 14, vertical: 10),
                            decoration: BoxDecoration(
                              color: t.surface,
                              borderRadius: BorderRadius.circular(12),
                              border: Border.all(color: t.border, width: 0.5),
                            ),
                            child: Row(
                              children: [
                                Expanded(
                                    child: Text(memory,
                                        style: TextStyle(
                                            color: t.textPrimary,
                                            fontSize: 14,
                                            height: 1.4))),
                                const SizedBox(width: 8),
                                GestureDetector(
                                  onTap: () => _delete(memory),
                                  child: Icon(Icons.close,
                                      color: t.textHint, size: 16),
                                ),
                              ],
                            ),
                          ),
                        );
                      },
                    ),
        );
      },
    );
  }
}

// ─── Settings Screen ──────────────────────────────────────────────────────────

class SettingsScreen extends StatefulWidget {
  const SettingsScreen({super.key});

  @override
  State<SettingsScreen> createState() => _SettingsScreenState();
}

class _SettingsScreenState extends State<SettingsScreen> {
  final TextEditingController _systemPromptController = TextEditingController();
  final TextEditingController _aboutMeController = TextEditingController();

  @override
  void initState() {
    super.initState();
    _loadSettings();
  }

  Future<void> _loadSettings() async {
    final prefs = await SharedPreferences.getInstance();
    setState(() {
      _systemPromptController.text = prefs.getString('systemPrompt') ?? '';
      _aboutMeController.text = prefs.getString('aboutMe') ?? '';
    });
  }

  Future<void> _saveSettings() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString('systemPrompt', _systemPromptController.text.trim());
    await prefs.setString('aboutMe', _aboutMeController.text.trim());
    // Reset install state so _loadModel performs a fresh install with new settings.
    _resetModelInstall();
    if (mounted) Navigator.pop(context, true);
  }

  @override
  void dispose() {
    _systemPromptController.dispose();
    _aboutMeController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return ValueListenableBuilder<bool>(
      valueListenable: isDarkMode,
      builder: (context, isDark, _) {
        final t = AppTheme(isDark);
        return Scaffold(
          backgroundColor: t.bg,
          appBar: AppBar(
            backgroundColor: t.bg,
            elevation: 0,
            leading: IconButton(
                icon: Icon(Icons.arrow_back, color: t.textSecond, size: 20),
                onPressed: () => Navigator.pop(context)),
            centerTitle: true,
            title: Text('Settings',
                style: TextStyle(
                    color: t.textPrimary, fontSize: 15, fontWeight: FontWeight.w500)),
            actions: [
              TextButton(
                  onPressed: _saveSettings,
                  child: Text('Save',
                      style: TextStyle(
                          color: t.textPrimary,
                          fontSize: 14,
                          fontWeight: FontWeight.w500))),
              const SizedBox(width: 8),
            ],
            bottom: PreferredSize(
                preferredSize: const Size.fromHeight(0.5),
                child: Container(height: 0.5, color: t.border)),
          ),
          body: ListView(
            padding: const EdgeInsets.all(20),
            children: [
              Text('AI PERSONALITY',
                  style: TextStyle(
                      color: t.textSecond,
                      fontSize: 12,
                      fontWeight: FontWeight.w500,
                      letterSpacing: 0.5)),
              const SizedBox(height: 8),
              Container(
                decoration: BoxDecoration(
                    color: t.surface,
                    borderRadius: BorderRadius.circular(12),
                    border: Border.all(color: t.border, width: 0.5)),
                child: TextField(
                  controller: _systemPromptController,
                  maxLines: 5,
                  style: TextStyle(color: t.textPrimary, fontSize: 14, height: 1.5),
                  decoration: InputDecoration(
                    hintText: 'e.g. You are a helpful assistant who speaks concisely.',
                    hintStyle: TextStyle(color: t.textHint, fontSize: 14),
                    border: InputBorder.none,
                    contentPadding: const EdgeInsets.all(14),
                  ),
                ),
              ),
              const SizedBox(height: 6),
              Text('Sets how the AI behaves and responds.',
                  style: TextStyle(color: t.textHint, fontSize: 12)),
              const SizedBox(height: 28),
              Text('ABOUT YOU',
                  style: TextStyle(
                      color: t.textSecond,
                      fontSize: 12,
                      fontWeight: FontWeight.w500,
                      letterSpacing: 0.5)),
              const SizedBox(height: 8),
              Container(
                decoration: BoxDecoration(
                    color: t.surface,
                    borderRadius: BorderRadius.circular(12),
                    border: Border.all(color: t.border, width: 0.5)),
                child: TextField(
                  controller: _aboutMeController,
                  maxLines: 5,
                  style: TextStyle(color: t.textPrimary, fontSize: 14, height: 1.5),
                  decoration: InputDecoration(
                    hintText:
                        'e.g. My name is Alex. I am a software developer who loves hiking.',
                    hintStyle: TextStyle(color: t.textHint, fontSize: 14),
                    border: InputBorder.none,
                    contentPadding: const EdgeInsets.all(14),
                  ),
                ),
              ),
              const SizedBox(height: 6),
              Text('The AI will remember this about you.',
                  style: TextStyle(color: t.textHint, fontSize: 12)),
            ],
          ),
        );
      },
    );
  }
}

// ─── Chat Screen ──────────────────────────────────────────────────────────────

class ChatScreen extends StatefulWidget {
  const ChatScreen({super.key});

  @override
  State<ChatScreen> createState() => _ChatScreenState();
}

class _ChatScreenState extends State<ChatScreen> with TickerProviderStateMixin {

  final List<Map<String, String>> _messages = [];
  final TextEditingController _controller = TextEditingController();
  final ScrollController _scrollController = ScrollController();
  final FocusNode _focusNode = FocusNode();
  final GlobalKey<ScaffoldState> _scaffoldKey = GlobalKey<ScaffoldState>();

  final SpeechToText _speech = SpeechToText();
  bool _speechAvailable = false;
  bool _isListening = false;

  bool _isLoading = true;
  bool _isThinking = false;
  bool _isSavingMemory = false;
  bool _isDownloading = false;    // true only during the first-launch model download
  double _downloadProgress = 0.0; // 0.0 – 1.0, drives the progress bar
  int _statementIndex = 0;        // drives rotating privacy statements
  Timer? _statementTimer;         // ticks every 4s to rotate statements
  String _loadingStatus = 'Starting...';
  InferenceChat? _chat;
  String? _currentChatName;
  List<SavedChat> _savedChats = [];
  List<String> _memories = [];

  // Guard flag to prevent overlapping sends (e.g. rapid double-tap)
  bool _isSending = false;

  // Throttle flag — prevents stacking hundreds of scroll animations during streaming.
  bool _scrollScheduled = false;

  late final AnimationController _dot1;
  late final AnimationController _dot2;
  late final AnimationController _dot3;
  late final AnimationController _micPulse;

  static const List<String> _allStarters = [
    'Help me write something',
    'Explain something simply',
    'Give me a step-by-step plan',
    'Help me think through a problem',
    'Summarize a topic for me',
    'Help me make a decision',
    'Give me a creative idea',
    'Write me a short story',
    'Help me brainstorm',
    'Give me a unique perspective',
    'Give me advice on something',
    'Quiz me on a topic',
    'Help me learn something new',
    'Give me a productivity tip',
    'Surprise me',
    'Tell me something fascinating',
    'Give me a fun challenge',
    'Tell me a joke',
  ];

  late final List<String> _visibleStarters = () {
    final short = _allStarters.where((s) => s.length <= 28).toList()..shuffle(Random());
    final long = _allStarters.where((s) => s.length > 28).toList()..shuffle(Random());
    return [...short, ...long].take(4).toList();
  }();

  @override
  void initState() {
    super.initState();
    _dot1 = AnimationController(vsync: this, duration: const Duration(milliseconds: 500))
      ..repeat(reverse: true);
    _dot2 = AnimationController(vsync: this, duration: const Duration(milliseconds: 500));
    _dot3 = AnimationController(vsync: this, duration: const Duration(milliseconds: 500));
    _micPulse = AnimationController(vsync: this, duration: const Duration(milliseconds: 800));
    Future.delayed(const Duration(milliseconds: 167),
        () { if (mounted) _dot2.repeat(reverse: true); });
    Future.delayed(const Duration(milliseconds: 334),
        () { if (mounted) _dot3.repeat(reverse: true); });
    _loadSavedChats();
    _loadMemories();
    _loadModel();
    _initSpeech();
  }

  @override
  void dispose() {
    _dot1.dispose();
    _dot2.dispose();
    _dot3.dispose();
    _micPulse.dispose();
    _statementTimer?.cancel();
    _controller.dispose();
    _scrollController.dispose();
    _focusNode.dispose();
    super.dispose();
  }

  // ─── Scroll helpers ──────────────────────────────────────────────────────────

  // Scrolls to the latest message. Throttled so rapid token callbacks during
  // streaming don't stack hundreds of competing 300ms animations causing jank.
  void _scrollToBottom() {
    if (_scrollScheduled) return;
    _scrollScheduled = true;
    WidgetsBinding.instance.addPostFrameCallback((_) {
      _scrollScheduled = false;
      if (_scrollController.hasClients) {
        _scrollController.animateTo(
          0,
          duration: const Duration(milliseconds: 200),
          curve: Curves.easeOut,
        );
      }
    });
  }

  // ─── Memory ──────────────────────────────────────────────────────────────────

  Future<void> _loadMemories() async {
    _memories = await MemoryManager.loadMemories();
    if (mounted) setState(() {});
  }

  // Extracts memories from conversation — call ONLY after _chat is closed.
  Future<void> _extractAndSaveMemories(List<Map<String, String>> messages) async {
    if (messages.length < 2) return;
    try {
      final userMessages = messages
          .where((m) => m['role'] == 'user')
          .map((m) => m['text'] ?? '')
          .where((t) => t.isNotEmpty)
          .take(10)
          .join('\n');

      if (userMessages.isEmpty) return;

      final appDir = await getApplicationDocumentsDirectory();
      final modelPath = '${appDir.path}/${_modelConfig.filename}';

      // Use shared install helper — avoids redundant reinstall.
      await _ensureModelInstalled(modelPath);

      final model = await FlutterGemma.getActiveModel(maxTokens: 100);
      final memChat = await model.createChat();
      await memChat.addQuery(Message(
        text: 'Extract up to 5 USEFUL personal facts about the user from these messages.\n'
              'ONLY save facts that would help personalize future responses:\n'
              '✓ Name, age, location, job, family\n'
              '✓ Specific interests, hobbies, preferences\n'
              '✓ Goals, problems they are working on\n'
              '✓ Strong opinions or values\n'
              'DO NOT save:\n'
              '✗ Communication style ("speaks English", "is polite")\n'
              '✗ Vague observations ("asks questions", "seeks information")\n'
              '✗ Obvious or trivial facts\n'
              'Reply with ONLY bullet points starting with "- ".\n'
              'If no useful facts exist, reply: NONE\n\n'
              '$userMessages',
        isUser: true,
      ));

      final stream = memChat.generateChatResponseAsync();
      final buffer = StringBuffer();
      int tokenCount = 0;

      await for (final response in stream) {
        if (response is TextResponse) {
          buffer.write(response.token);
          tokenCount++;
          // Hard stop at 150 tokens to prevent infinite loops.
          if (tokenCount >= 150) break;
        }
      }

      await memChat.close();

      final result = buffer.toString().trim();
      if (result.isEmpty || result.toUpperCase().contains('NONE')) return;

      final newMemories = result
          .split('\n')
          .where((line) => line.trim().startsWith('-'))
          .map((line) => line.trim().replaceFirst(RegExp(r'^-+\s*'), '').trim())
          .where((line) => line.isNotEmpty && line.length > 10)
          // Reject vague behavioral observations the model occasionally still saves.
          .where((line) {
            final lower = line.toLowerCase();
            return !lower.contains('communicates in') &&
                   !lower.contains('speaks english') &&
                   !lower.contains('is polite') &&
                   !lower.contains('asks questions') &&
                   !lower.contains('seeks information') &&
                   !lower.contains('seeking information') &&
                   !lower.contains('looking for') &&
                   !lower.contains('the user is communicating') &&
                   !lower.contains('uses english');
          })
          .take(5)
          .toList();

      if (newMemories.isNotEmpty) {
        await MemoryManager.addMemories(newMemories);
        _memories = await MemoryManager.loadMemories();
        if (mounted) setState(() {});
      }
    } catch (e) {
      // Silent fail — memory extraction is best-effort.
    }
  }

  // ─── Speech ──────────────────────────────────────────────────────────────────

  Future<void> _initSpeech() async {
    _speechAvailable = await _speech.initialize(
      onError: (error) {
        if (mounted) setState(() => _isListening = false);
        _micPulse.stop();
        _micPulse.reset();
      },
      onStatus: (status) {
        if (status == 'done' || status == 'notListening') {
          if (mounted) setState(() => _isListening = false);
          _micPulse.stop();
          _micPulse.reset();
        }
      },
    );
    if (mounted) setState(() {});
  }

  Future<void> _toggleListening() async {
    if (!_speechAvailable) {
      _showSnackBar('Microphone not available');
      return;
    }
    if (_isListening) {
      await _speech.stop();
      _micPulse.stop();
      _micPulse.reset();
      if (mounted) setState(() => _isListening = false);
    } else {
      HapticFeedback.mediumImpact();
      if (mounted) setState(() => _isListening = true);
      _micPulse.repeat(reverse: true);
      await _speech.listen(
        onResult: (result) {
          if (mounted) {
            setState(() {
              _controller.text = result.recognizedWords;
              _controller.selection = TextSelection.fromPosition(
                  TextPosition(offset: _controller.text.length));
            });
          }
          if (result.finalResult && result.recognizedWords.isNotEmpty) {
            _micPulse.stop();
            _micPulse.reset();
            if (mounted) setState(() => _isListening = false);
            Future.delayed(const Duration(milliseconds: 300), _sendMessage);
          }
        },
        listenFor: const Duration(seconds: 30),
        pauseFor: const Duration(seconds: 3),
        localeId: 'en_US',
      );
    }
  }

  // ─── Saved Chats ─────────────────────────────────────────────────────────────

  Future<void> _loadSavedChats() async {
    final prefs = await SharedPreferences.getInstance();
    final raw = prefs.getStringList('savedChats') ?? [];
    if (mounted) {
      setState(() {
        _savedChats = raw
            .map((s) => SavedChat.fromJson(jsonDecode(s)))
            .toList()
          ..sort((a, b) => b.savedAt.compareTo(a.savedAt));
      });
    }
  }

  Future<void> _persistSavedChats() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setStringList(
        'savedChats', _savedChats.map((c) => jsonEncode(c.toJson())).toList());
  }

  Future<void> _saveCurrentChat() async {
    if (_messages.isEmpty) {
      _showSnackBar('Nothing to save yet.');
      return;
    }
    // Guard: don't attempt name generation while the main chat session is active —
    // creating a second model session concurrently risks corrupting the live one.
    if (_isThinking) {
      _showSnackBar('Wait for the response to finish before saving.');
      return;
    }
    final snapshot = List<Map<String, String>>.from(_messages);
    _currentChatName ??= await _generateChatName(snapshot);
    final name = _currentChatName ?? 'Chat';
    final chat = SavedChat(
      id: DateTime.now().millisecondsSinceEpoch.toString(),
      name: name,
      messages: snapshot,
      savedAt: DateTime.now(),
    );
    setState(() => _savedChats.insert(0, chat));
    await _persistSavedChats();
    _showSnackBar('Saved as "$name"');
  }

  Future<String> _generateChatName(List<Map<String, String>> messages) async {
    try {
      final summary = messages
          .where((m) => m['role'] == 'user')
          .take(2)
          .map((m) => m['text'])
          .join(' ');
      final appDir = await getApplicationDocumentsDirectory();
      final modelPath = '${appDir.path}/${_modelConfig.filename}';
      await _ensureModelInstalled(modelPath);
      final model = await FlutterGemma.getActiveModel(maxTokens: 64);
      final nameChat = await model.createChat();
      await nameChat.addQuery(Message(
          text: 'Give this conversation a title of 3-5 words. Reply with ONLY the title, nothing else. Conversation: $summary',
          isUser: true));
      final stream = nameChat.generateChatResponseAsync();
      final buffer = StringBuffer();
      int tokenCount = 0;
      await for (final response in stream) {
        if (response is TextResponse) {
          buffer.write(response.token);
          tokenCount++;
          // Hard stop — model occasionally rambles on name generation too.
          if (tokenCount >= 30) break;
        }
      }
      await nameChat.close();
      final name = buffer.toString().trim().replaceAll('"', '').replaceAll("'", '');
      return name.isNotEmpty ? name : 'Saved Chat';
    } catch (_) {
      return 'Saved Chat';
    }
  }

  Future<void> _loadSavedChat(SavedChat savedChat) async {
    // Guard: loading a saved chat while the AI is streaming would leave the old
    // stream writing tokens into the newly loaded message list.
    if (_isThinking) {
      Navigator.pop(context);
      _showSnackBar('Wait for the response to finish first.');
      return;
    }
    Navigator.pop(context);
    try {
      final freshChat = await _createFreshChat();
      if (mounted) {
        setState(() {
          _chat = freshChat;
          _messages.clear();
          _messages.addAll(savedChat.messages);
          _currentChatName = savedChat.name;
        });
        _scrollToBottom();
      }
    } catch (e) {
      _showSnackBar('Failed to load chat.');
    }
  }

  Future<void> _deleteSavedChat(String id) async {
    setState(() => _savedChats.removeWhere((c) => c.id == id));
    await _persistSavedChats();
  }

  // ─── Model ───────────────────────────────────────────────────────────────────

  Future<void> _downloadModel(String url, String savePath) async {
    final client = http.Client();
    final file = File(savePath);
    IOSink? sink;
    try {
      final authRequest = http.Request('GET', Uri.parse(url));
      // NOTE: Replace with your actual HuggingFace token before building.
     authRequest.headers['Authorization'] = 'Bearer YOUR_HUGGINGFACE_TOKEN_HERE';
      authRequest.followRedirects = false;
      final authResponse = await client.send(authRequest);
      final redirectUrl = authResponse.headers['location'];
      if (redirectUrl == null) throw Exception('No redirect URL received');
      await authResponse.stream.drain();
      final fileRequest = http.Request('GET', Uri.parse(redirectUrl));
      final fileResponse = await client.send(fileRequest);
      if (fileResponse.statusCode != 200) {
        throw Exception('Download failed: HTTP ${fileResponse.statusCode}');
      }
      final totalBytes = fileResponse.contentLength ?? 0;
      var received = 0;
      sink = file.openWrite();
      await for (final chunk in fileResponse.stream) {
        sink.add(chunk);
        received += chunk.length;
        if (totalBytes > 0 && mounted) {
          setState(() {
            _downloadProgress = received / totalBytes;
            // Keep _loadingStatus in sync for accessibility / fallback text.
            _loadingStatus =
                'Downloading — ${(_downloadProgress * 100).toStringAsFixed(1)}%';
          });
        }
      }
      await sink.close();
      sink = null;
    } catch (e) {
      // Always close the sink and delete the partial file so the next launch
      // doesn't find a corrupt file that passes the existence check but fails to load.
      await sink?.close();
      if (await file.exists()) await file.delete();
      rethrow;
    } finally {
      client.close();
    }
  }

  Future<void> _loadModel() async {
    try {
      final appDir = await getApplicationDocumentsDirectory();
      final modelPath = '${appDir.path}/${_modelConfig.filename}';
      if (!await File(modelPath).exists()) {
        setState(() {
          _isDownloading = true;
          _downloadProgress = 0.0;
          _statementIndex = 0;
          _loadingStatus = 'Starting download...';
        });
        // Rotate privacy statements every 4 seconds, independent of download speed.
        _statementTimer = Timer.periodic(const Duration(seconds: 4), (_) {
          if (mounted) setState(() => _statementIndex = (_statementIndex + 1) % 5);
        });
        await _downloadModel(_modelConfig.downloadUrl, modelPath);
        _statementTimer?.cancel();
        _statementTimer = null;
        setState(() => _isDownloading = false);
      }
      setState(() => _loadingStatus = 'Loading model...');

      // Always do a fresh install on explicit _loadModel calls (startup + settings change).
      _resetModelInstall();
      await _ensureModelInstalled(modelPath);

      final prefs = await SharedPreferences.getInstance();
      final systemPrompt = prefs.getString('systemPrompt') ?? '';
      final aboutMe = prefs.getString('aboutMe') ?? '';
      final model = await FlutterGemma.getActiveModel(maxTokens: _modelConfig.maxTokens);
      _chat = await model.createChat(
          systemInstruction: _buildSystemInstruction(systemPrompt, aboutMe));
      if (mounted) setState(() => _isLoading = false);
    } catch (e) {
      _statementTimer?.cancel();
      _statementTimer = null;
      if (mounted) setState(() {
        _isDownloading = false;
        _loadingStatus = 'Error: $e';
      });
    }
  }

  String _buildSystemInstruction(String systemPrompt, String aboutMe) {
    final parts = <String>[];

    // User's custom personality prompt (editable in Settings).
    if (systemPrompt.isNotEmpty) parts.add(systemPrompt);

    // Immutable core layer — always present underneath user customization.
    parts.add(
      'Always be concise on mobile. Never use emojis. Engage genuinely with questions about preferences or feelings — '
      'share perspectives rather than deflecting with "I\'m just an AI". '
      'Match the user\'s tone naturally.'
    );

    if (aboutMe.isNotEmpty) parts.add('About the user: $aboutMe');
    final memoryPrompt = MemoryManager.buildMemoryPrompt(_memories);
    if (memoryPrompt.isNotEmpty) parts.add(memoryPrompt);
    return parts.join('\n\n');
  }

  Future<InferenceChat> _createFreshChat() async {
    await _chat?.close();
    _chat = null;
    final appDir = await getApplicationDocumentsDirectory();
    final modelPath = '${appDir.path}/${_modelConfig.filename}';
    await _ensureModelInstalled(modelPath);
    final prefs = await SharedPreferences.getInstance();
    final systemPrompt = prefs.getString('systemPrompt') ?? '';
    final aboutMe = prefs.getString('aboutMe') ?? '';
    // Load fresh memories so this session has the latest.
    _memories = await MemoryManager.loadMemories();
    final model = await FlutterGemma.getActiveModel(maxTokens: _modelConfig.maxTokens);
    // Use _buildSystemInstruction so the immutable core layer is always present.
    return await model.createChat(
        systemInstruction: _buildSystemInstruction(systemPrompt, aboutMe));
  }

  // ─── Actions ─────────────────────────────────────────────────────────────────

  Future<void> _sendMessage() async {
    final text = _controller.text.trim();
    // FIX: Added _isSending guard to prevent overlapping sends on rapid taps.
    if (text.isEmpty || _isThinking || _isSending || _chat == null) return;
    HapticFeedback.lightImpact();
    _isSending = true;
    setState(() {
      _messages.add({'role': 'user', 'text': text});
      _messages.add({'role': 'ai', 'text': ''});
      _isThinking = true;
      _controller.clear();
    });
    // FIX: Scroll to bottom so the new message is visible immediately.
    _scrollToBottom();
    try {
      await _chat!.addQuery(Message(text: text, isUser: true));
      final stream = _chat!.generateChatResponseAsync();
      await for (final response in stream) {
        if (response is TextResponse && mounted) {
          setState(() {
            _messages.last['text'] = (_messages.last['text'] ?? '') + response.token;
          });
          // FIX: Keep scrolled to bottom while streaming response tokens.
          _scrollToBottom();
        }
      }
      HapticFeedback.selectionClick();
    } catch (e) {
      if (mounted) setState(() => _messages.last['text'] = 'Error: $e');
    } finally {
      if (mounted) setState(() => _isThinking = false);
      _isSending = false;
    }
  }

  Future<void> _regenerateLastResponse() async {
    if (_isThinking || _chat == null) return;
    final lastUserIndex = _messages.lastIndexWhere((m) => m['role'] == 'user');
    if (lastUserIndex == -1) return;
    final lastUserText = _messages[lastUserIndex]['text'] ?? '';
    HapticFeedback.mediumImpact();
    setState(() {
      _messages.removeRange(lastUserIndex + 1, _messages.length);
      _messages.add({'role': 'ai', 'text': ''});
      _isThinking = true;
    });
    _scrollToBottom();
    try {
      await _chat!.addQuery(Message(text: lastUserText, isUser: true));
      final stream = _chat!.generateChatResponseAsync();
      await for (final response in stream) {
        if (response is TextResponse && mounted) {
          setState(() {
            _messages.last['text'] = (_messages.last['text'] ?? '') + response.token;
          });
          _scrollToBottom();
        }
      }
      HapticFeedback.selectionClick();
    } catch (e) {
      if (mounted) setState(() => _messages.last['text'] = 'Error: $e');
    } finally {
      if (mounted) setState(() => _isThinking = false);
    }
  }

  void _showMessageMenu(BuildContext context, int index, AppTheme t) {
    final msg = _messages[index];
    final isUser = msg['role'] == 'user';
    final text = msg['text'] ?? '';
    HapticFeedback.mediumImpact();
    showModalBottomSheet(
      context: context,
      backgroundColor: t.surface,
      shape: const RoundedRectangleBorder(
          borderRadius: BorderRadius.vertical(top: Radius.circular(20))),
      builder: (ctx) => SafeArea(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Container(
                width: 36,
                height: 4,
                margin: const EdgeInsets.only(top: 12, bottom: 8),
                decoration: BoxDecoration(
                    color: t.border, borderRadius: BorderRadius.circular(2))),
            ListTile(
              leading: Icon(Icons.copy_outlined, color: t.textSecond, size: 20),
              title: Text('Copy', style: TextStyle(color: t.textPrimary, fontSize: 14)),
              onTap: () {
                Navigator.pop(ctx);
                _copyMessage(text);
              },
            ),
            if (!isUser)
              ListTile(
                leading:
                    Icon(Icons.refresh_rounded, color: t.textSecond, size: 20),
                title: Text('Regenerate',
                    style: TextStyle(color: t.textPrimary, fontSize: 14)),
                onTap: () {
                  Navigator.pop(ctx);
                  _regenerateLastResponse();
                },
              ),
            const SizedBox(height: 8),
          ],
        ),
      ),
    );
  }

  void _newChat() {
    // FIX: Don't show dialog if there's nothing to lose.
    if (_messages.isEmpty) return;

    showDialog(
      context: context,
      builder: (ctx) {
        final t = AppTheme(isDarkMode.value);
        return AlertDialog(
          backgroundColor: t.surface,
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
          title: Text('New chat',
              style: TextStyle(
                  color: t.textPrimary, fontSize: 16, fontWeight: FontWeight.w500)),
          content: Text('Unsaved messages will be lost. Continue?',
              style: TextStyle(color: t.textSecond, fontSize: 14, height: 1.4)),
          actions: [
            TextButton(
                onPressed: () => Navigator.pop(ctx),
                child: Text('Cancel', style: TextStyle(color: t.textSecond))),
            TextButton(
              onPressed: () async {
                Navigator.pop(ctx);
                try {
                  // Snapshot messages before clearing.
                  final snapshot = List<Map<String, String>>.from(_messages);

                  // Clear UI immediately — no flash of old messages.
                  if (mounted) {
                    setState(() {
                      _isSavingMemory = true;
                      _messages.clear();
                      _currentChatName = null;
                    });
                  }

                  // Close old session FIRST — critical ordering.
                  await _chat?.close();
                  _chat = null;

                  // Extract memories from snapshot (session is closed — safe).
                  await _extractAndSaveMemories(snapshot);

                  // Create fresh chat with updated memories.
                  final freshChat = await _createFreshChat();
                  if (mounted) {
                    setState(() {
                      _chat = freshChat;
                      _isSavingMemory = false;
                    });
                  }
                } catch (_) {
                  if (mounted) setState(() => _isSavingMemory = false);
                }
              },
              child: const Text('Continue', style: TextStyle(color: AppColors.danger)),
            ),
          ],
        );
      },
    );
  }

  void _copyMessage(String text) {
    Clipboard.setData(ClipboardData(text: text));
    _showSnackBar('Copied');
  }

  void _showSnackBar(String message) {
    final t = AppTheme(isDarkMode.value);
    ScaffoldMessenger.of(context).showSnackBar(SnackBar(
      content: Text(message, style: TextStyle(color: t.textPrimary, fontSize: 13)),
      backgroundColor: t.surfaceHigh,
      behavior: SnackBarBehavior.floating,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
      duration: const Duration(seconds: 2),
    ));
  }

  Future<void> _launchUrl(String url) async {
    final uri = Uri.parse(url);
    if (await canLaunchUrl(uri)) {
      await launchUrl(uri, mode: LaunchMode.externalApplication);
    }
  }

  void _openSettings() async {
    Navigator.pop(context);
    final result = await Navigator.push(
        context,
        PageRouteBuilder(
          pageBuilder: (_, animation, _) => const SettingsScreen(),
          transitionsBuilder: (_, animation, _x, child) => SlideTransition(
            position: Tween<Offset>(begin: const Offset(1, 0), end: Offset.zero)
                .animate(CurvedAnimation(parent: animation, curve: Curves.easeOutCubic)),
            child: child,
          ),
        ));
    if (result == true && mounted) {
      setState(() {
        _isLoading = true;
        _loadingStatus = 'Applying settings...';
      });
      await _loadModel();
    }
  }

  void _openMemory() async {
    Navigator.pop(context);
    final memoriesChanged = await Navigator.push<bool>(
        context,
        PageRouteBuilder(
          pageBuilder: (_, animation, _x) => const MemoryScreen(),
          transitionsBuilder: (_, animation, _x, child) => SlideTransition(
            position: Tween<Offset>(begin: const Offset(1, 0), end: Offset.zero)
                .animate(CurvedAnimation(parent: animation, curve: Curves.easeOutCubic)),
            child: child,
          ),
        ));
    // Always refresh the badge count.
    await _loadMemories();
    // If memories were deleted, rebuild the chat session immediately so the
    // current session doesn't retain stale memories in its system prompt.
    if (memoriesChanged == true && _chat != null && !_isThinking) {
      final freshChat = await _createFreshChat();
      if (mounted) setState(() => _chat = freshChat);
    }
  }

  // ─── Drawer ──────────────────────────────────────────────────────────────────

  Widget _buildDrawer(AppTheme t) {
    return Drawer(
      backgroundColor: t.surface,
      child: SafeArea(
        child: Column(
          children: [
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 24),
              child: Row(children: [
                Container(
                  width: 36,
                  height: 36,
                  decoration: BoxDecoration(
                      color: t.surfaceHigh,
                      borderRadius: BorderRadius.circular(10),
                      border: Border.all(color: t.border, width: 0.5)),
                  child: Center(child: VeilIcon(size: 22, isDark: t.isDark)),
                ),
                const SizedBox(width: 12),
                Text('Veil',
                    style: TextStyle(
                        color: t.textPrimary,
                        fontSize: 16,
                        fontWeight: FontWeight.w600)),
              ]),
            ),
            Divider(color: t.border, height: 1),
            const SizedBox(height: 8),
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 4),
              child: Row(children: [
                Text('SAVED CHATS',
                    style: TextStyle(
                        color: t.textHint,
                        fontSize: 11,
                        fontWeight: FontWeight.w500,
                        letterSpacing: 0.8)),
              ]),
            ),
            Expanded(
              child: _savedChats.isEmpty
                  ? Center(
                      child: Text('No saved chats yet',
                          style: TextStyle(color: t.textHint, fontSize: 13)))
                  : ListView.builder(
                      padding: const EdgeInsets.symmetric(horizontal: 12),
                      itemCount: _savedChats.length,
                      itemBuilder: (context, index) {
                        final chat = _savedChats[index];
                        return Dismissible(
                          key: Key(chat.id),
                          direction: DismissDirection.endToStart,
                          onDismissed: (_) {
                            HapticFeedback.mediumImpact();
                            _deleteSavedChat(chat.id);
                          },
                          background: Container(
                            alignment: Alignment.centerRight,
                            padding: const EdgeInsets.only(right: 16),
                            decoration: BoxDecoration(
                                color: AppColors.danger.withValues(alpha: 0.1),
                                borderRadius: BorderRadius.circular(10)),
                            child: const Icon(Icons.delete_outline,
                                color: AppColors.danger, size: 18),
                          ),
                          child: ListTile(
                            contentPadding:
                                const EdgeInsets.symmetric(horizontal: 8),
                            leading: Icon(Icons.chat_bubble_outline,
                                color: t.textHint, size: 16),
                            title: Text(chat.name,
                                style: TextStyle(
                                    color: t.textPrimary,
                                    fontSize: 13,
                                    fontWeight: FontWeight.w400),
                                maxLines: 1,
                                overflow: TextOverflow.ellipsis),
                            onTap: () => _loadSavedChat(chat),
                            shape: RoundedRectangleBorder(
                                borderRadius: BorderRadius.circular(10)),
                          ),
                        );
                      },
                    ),
            ),
            Divider(color: t.border, height: 1),
            // Memory row
            Padding(
              padding: const EdgeInsets.fromLTRB(12, 4, 12, 0),
              child: ListTile(
                contentPadding: const EdgeInsets.symmetric(horizontal: 8),
                leading: Icon(Icons.psychology_outlined,
                    color: t.textSecond, size: 18),
                title: Text('Memory',
                    style: TextStyle(color: t.textPrimary, fontSize: 14)),
                trailing: _memories.isNotEmpty
                    ? Container(
                        padding: const EdgeInsets.symmetric(
                            horizontal: 7, vertical: 2),
                        decoration: BoxDecoration(
                            color: t.surfaceHigh,
                            borderRadius: BorderRadius.circular(10)),
                        child: Text('${_memories.length}',
                            style:
                                TextStyle(color: t.textSecond, fontSize: 11)),
                      )
                    : null,
                onTap: _openMemory,
                shape:
                    RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
              ),
            ),
            Divider(color: t.border, height: 1),
            // Dark mode toggle
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 12),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Row(children: [
                    Icon(
                        isDarkMode.value
                            ? Icons.dark_mode_outlined
                            : Icons.light_mode_outlined,
                        color: t.textSecond,
                        size: 18),
                    const SizedBox(width: 12),
                    Text('Dark mode',
                        style: TextStyle(color: t.textPrimary, fontSize: 14)),
                  ]),
                  Switch(
                    value: isDarkMode.value,
                    onChanged: (val) async {
                      HapticFeedback.selectionClick();
                      isDarkMode.value = val;
                      final prefs = await SharedPreferences.getInstance();
                      await prefs.setBool('isDark', val);
                      if (mounted) setState(() {});
                    },
                    thumbColor: WidgetStateProperty.resolveWith((states) =>
                        states.contains(WidgetState.selected)
                            ? t.textPrimary
                            : t.textHint),
                    trackColor: WidgetStateProperty.resolveWith((states) =>
                        states.contains(WidgetState.selected)
                            ? t.textSecond
                            : t.surfaceHigh),
                  ),
                ],
              ),
            ),
            // Support row
            Padding(
              padding: const EdgeInsets.fromLTRB(12, 0, 12, 0),
              child: ListTile(
                contentPadding: const EdgeInsets.symmetric(horizontal: 8),
                leading:
                    Icon(Icons.coffee_outlined, color: t.textSecond, size: 18),
                title: Text('Support Veil',
                    style: TextStyle(color: t.textPrimary, fontSize: 14)),
                onTap: () {
                  Navigator.pop(context);
                  _launchUrl('https://ko-fi.com/VeilAIapp');
                },
                shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(10)),
              ),
            ),
            // Settings row
            Padding(
              padding: const EdgeInsets.fromLTRB(12, 0, 12, 12),
              child: ListTile(
                contentPadding: const EdgeInsets.symmetric(horizontal: 8),
                leading:
                    Icon(Icons.settings_outlined, color: t.textSecond, size: 18),
                title: Text('Settings',
                    style: TextStyle(color: t.textPrimary, fontSize: 14)),
                onTap: _openSettings,
                shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(10)),
              ),
            ),
          ],
        ),
      ),
    );
  }

  // ─── Widgets ─────────────────────────────────────────────────────────────────

  Widget _buildDot(AnimationController controller, AppTheme t) {
    return AnimatedBuilder(
      animation: controller,
      builder: (_, _) => Transform.translate(
        offset: Offset(0, -3 * controller.value),
        child: Container(
            width: 5,
            height: 5,
            decoration:
                BoxDecoration(color: t.textSecond, shape: BoxShape.circle)),
      ),
    );
  }

  Widget _buildTypingDots(AppTheme t) {
    return Row(mainAxisSize: MainAxisSize.min, children: [
      _buildDot(_dot1, t),
      const SizedBox(width: 5),
      _buildDot(_dot2, t),
      const SizedBox(width: 5),
      _buildDot(_dot3, t),
    ]);
  }

  Widget _buildLoadingScreen(AppTheme t) {
    if (_isDownloading) {
      return _buildDownloadScreen(t);
    }
    // Standard loading (startup after first launch, or applying settings).
    return Center(
        child: Column(mainAxisAlignment: MainAxisAlignment.center, children: [
      VeilIcon(size: 48, isDark: t.isDark),
      const SizedBox(height: 32),
      SizedBox(
          width: 22,
          height: 22,
          child: CircularProgressIndicator(
              color: t.textSecond, strokeWidth: 1.5)),
      const SizedBox(height: 20),
      Padding(
          padding: const EdgeInsets.symmetric(horizontal: 40),
          child: Text(_loadingStatus,
              textAlign: TextAlign.center,
              style:
                  TextStyle(fontSize: 13, color: t.textSecond, height: 1.5))),
      const SizedBox(height: 6),
      Text('On-device · Private',
          style: TextStyle(fontSize: 11, color: t.textHint)),
    ]));
  }

  // Shown only on first launch while the model downloads.
  // Designed to hold attention and reframe the wait as meaningful.
  Widget _buildDownloadScreen(AppTheme t) {
    const statements = [
      'The AI runs entirely on your phone.\nNo servers ever see your words.',
      'Your conversations are never stored\nor transmitted anywhere.',
      'No account. No tracking.\nNo data leaving your device.',
      'Most AI reads everything you type.\nVeil never can.',
      'Your thoughts are yours.\nThat\'s not a feature — it\'s a promise.',
    ];

    final pct = (_downloadProgress * 100).toStringAsFixed(0);

    return Container(
      // Explicit background prevents any bleed-through behind the column.
      color: t.bg,
      child: Center(
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 40),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              VeilIcon(size: 56, isDark: t.isDark),
              const SizedBox(height: 28),

              // Title
              Text('Setting up Veil',
                  style: TextStyle(
                      color: t.textPrimary,
                      fontSize: 18,
                      fontWeight: FontWeight.w300,
                      letterSpacing: 0.5)),
              const SizedBox(height: 8),
              Text(
                  'Downloading your private AI (${_modelConfig.displaySize})',
                  style: TextStyle(color: t.textSecond, fontSize: 13),
                  textAlign: TextAlign.center),

              const SizedBox(height: 36),

              // Progress bar + percentage
              Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      Text('Downloading',
                          style:
                              TextStyle(color: t.textHint, fontSize: 11)),
                      Text('$pct%',
                          style: TextStyle(
                              color: t.textSecond,
                              fontSize: 11,
                              fontWeight: FontWeight.w500)),
                    ],
                  ),
                  const SizedBox(height: 8),
                  ClipRRect(
                    borderRadius: BorderRadius.circular(4),
                    child: LinearProgressIndicator(
                      value: _downloadProgress,
                      minHeight: 3,
                      backgroundColor: t.surfaceHigh,
                      valueColor:
                          AlwaysStoppedAnimation<Color>(t.textPrimary),
                    ),
                  ),
                ],
              ),

              const SizedBox(height: 40),

              // Rotating privacy statements — driven by timer, not progress.
              // Fades between statements every 4 seconds regardless of speed.
              AnimatedSwitcher(
                duration: const Duration(milliseconds: 600),
                child: Text(
                  statements[_statementIndex],
                  key: ValueKey(_statementIndex),
                  textAlign: TextAlign.center,
                  style: TextStyle(
                      color: t.textSecond,
                      fontSize: 14,
                      height: 1.6,
                      fontWeight: FontWeight.w400), // w400 — crisp on all screens
                ),
              ),

              const SizedBox(height: 24),

              Text('This only happens once.',
                  style: TextStyle(
                      color: t.textHint,
                      fontSize: 12,
                      fontStyle: FontStyle.italic)),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildEmptyState(AppTheme t) {
    return Center(
        child: Column(mainAxisAlignment: MainAxisAlignment.center, children: [
      VeilIcon(size: 56, isDark: t.isDark),
      const SizedBox(height: 16),
      Text('VEIL',
          style: TextStyle(
              color: t.textHint,
              fontSize: 11,
              fontWeight: FontWeight.w500,
              letterSpacing: 4)),
    ]));
  }

  Widget _buildSavingMemoryState(AppTheme t) {
    return Column(
      mainAxisAlignment: MainAxisAlignment.end,
      children: [
        Padding(
          padding: const EdgeInsets.fromLTRB(16, 8, 52, 16),
          child: Align(
            alignment: Alignment.centerLeft,
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
              decoration: BoxDecoration(
                color: t.aiBubble,
                borderRadius: const BorderRadius.only(
                  topLeft: Radius.circular(16),
                  topRight: Radius.circular(16),
                  bottomRight: Radius.circular(16),
                  bottomLeft: Radius.circular(4),
                ),
                border: Border.all(color: t.border, width: 0.5),
              ),
              child: Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Icon(Icons.psychology_outlined, color: t.textHint, size: 14),
                  const SizedBox(width: 6),
                  Text('Saving memories...',
                      style: TextStyle(color: t.textHint, fontSize: 13)),
                ],
              ),
            ),
          ),
        ),
      ],
    );
  }

  Widget _buildMessage(int index, AppTheme t) {
    final msg = _messages[index];
    final isUser = msg['role'] == 'user';
    final text = msg['text'] ?? '';
    final isLastMessage = index == _messages.length - 1;
    final showTyping = !isUser && text.isEmpty && _isThinking && isLastMessage;

    return TweenAnimationBuilder<double>(
      tween: Tween(begin: 0, end: 1),
      duration: const Duration(milliseconds: 300),
      curve: Curves.easeOutCubic,
      builder: (context, value, child) => Opacity(
          opacity: value,
          child: Transform.translate(
              offset: Offset(0, 12 * (1 - value)), child: child)),
      child: Padding(
        padding: EdgeInsets.only(left: isUser ? 52 : 0, right: isUser ? 0 : 52),
        child: Align(
          alignment: isUser ? Alignment.centerRight : Alignment.centerLeft,
          child: GestureDetector(
            onLongPress: text.isNotEmpty
                ? () => _showMessageMenu(context, index, t)
                : null,
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
              decoration: BoxDecoration(
                color: isUser ? t.userBubble : t.aiBubble,
                borderRadius: BorderRadius.only(
                  topLeft: const Radius.circular(16),
                  topRight: const Radius.circular(16),
                  bottomLeft: Radius.circular(isUser ? 16 : 4),
                  bottomRight: Radius.circular(isUser ? 4 : 16),
                ),
                border: Border.all(color: t.border, width: 0.5),
              ),
              child: showTyping
                  ? Padding(
                      padding: const EdgeInsets.symmetric(
                          vertical: 3, horizontal: 2),
                      child: _buildTypingDots(t))
                  : isUser
                      ? Text(text,
                          style: TextStyle(
                              color: t.textPrimary,
                              fontSize: 15,
                              height: 1.5,
                              fontWeight: FontWeight.w400))
                      : MarkdownBody(
                          data: text,
                          styleSheet: MarkdownStyleSheet(
                            p: TextStyle(
                                color: t.textPrimary,
                                fontSize: 15,
                                height: 1.5,
                                fontWeight: FontWeight.w400),
                            code: TextStyle(
                                color: t.textPrimary,
                                fontSize: 13,
                                fontFamily: 'monospace',
                                backgroundColor: t.surfaceHigh),
                            codeblockDecoration: BoxDecoration(
                                color: t.surfaceHigh,
                                borderRadius: BorderRadius.circular(8)),
                            blockquote:
                                TextStyle(color: t.textSecond, fontSize: 14),
                            strong: TextStyle(
                                color: t.textPrimary,
                                fontWeight: FontWeight.w600),
                            em: TextStyle(
                                color: t.textPrimary,
                                fontStyle: FontStyle.italic),
                            listBullet:
                                TextStyle(color: t.textSecond, fontSize: 15),
                          ),
                        ),
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildMessageList(AppTheme t) {
    if (_isSavingMemory) return _buildSavingMemoryState(t);
    if (_messages.isEmpty) return _buildEmptyState(t);
    return ListView.separated(
      controller: _scrollController,
      reverse: true,
      padding: const EdgeInsets.fromLTRB(16, 16, 16, 8),
      itemCount: _messages.length,
      separatorBuilder: (_, _) => const SizedBox(height: 8),
      itemBuilder: (context, index) =>
          _buildMessage(_messages.length - 1 - index, t),
    );
  }

  Widget _buildInputArea(AppTheme t) {
    final bool showSend = _controller.text.isNotEmpty || _isThinking;
    final bool showStarters =
        _messages.isEmpty && !_isThinking && !_isSavingMemory;

    return Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        if (showStarters)
          Container(
            color: t.bg,
            padding: const EdgeInsets.fromLTRB(16, 12, 16, 0),
            child: Wrap(
              spacing: 8,
              runSpacing: 8,
              alignment: WrapAlignment.center,
              children: _visibleStarters.map((starter) {
                return GestureDetector(
                  onTap: () {
                    HapticFeedback.lightImpact();
                    _controller.text = starter;
                    setState(() {});
                    Future.delayed(
                        const Duration(milliseconds: 100), _sendMessage);
                  },
                  child: Container(
                    padding:
                        const EdgeInsets.symmetric(horizontal: 14, vertical: 8),
                    decoration: BoxDecoration(
                        color: t.surface,
                        borderRadius: BorderRadius.circular(20),
                        border: Border.all(color: t.border, width: 0.5)),
                    child: Text(starter,
                        style: TextStyle(
                            color: t.textSecond,
                            fontSize: 13,
                            fontWeight: FontWeight.w400)),
                  ),
                );
              }).toList(),
            ),
          ),
        Container(
          padding: const EdgeInsets.fromLTRB(16, 10, 16, 24),
          decoration: BoxDecoration(
              color: t.bg,
              border: Border(top: BorderSide(color: t.border, width: 0.5))),
          child: Row(
            crossAxisAlignment: CrossAxisAlignment.end,
            children: [
              Expanded(
                child: Container(
                  constraints: const BoxConstraints(maxHeight: 120),
                  decoration: BoxDecoration(
                    color: t.surface,
                    borderRadius: BorderRadius.circular(12),
                    border: Border.all(
                        color: _isListening ? t.textSecond : t.border,
                        width: _isListening ? 1 : 0.5),
                  ),
                  child: TextField(
                    controller: _controller,
                    focusNode: _focusNode,
                    maxLines: null,
                    minLines: 1,
                    enabled: !_isThinking && !_isSavingMemory,
                    onChanged: (_) => setState(() {}),
                    style: TextStyle(
                        color: t.textPrimary, fontSize: 15, height: 1.4),
                    decoration: InputDecoration(
                      hintText: _isSavingMemory
                          ? 'Saving memories...'
                          : _isListening
                              ? 'Listening...'
                              : _isThinking
                                  ? 'Thinking...'
                                  : 'Message',
                      hintStyle: TextStyle(
                          color: _isListening ? t.textSecond : t.textHint,
                          fontSize: 15),
                      border: InputBorder.none,
                      contentPadding: const EdgeInsets.symmetric(
                          horizontal: 14, vertical: 10),
                    ),
                    onSubmitted: (_) => _sendMessage(),
                  ),
                ),
              ),
              const SizedBox(width: 10),
              AnimatedSwitcher(
                duration: const Duration(milliseconds: 200),
                transitionBuilder: (child, animation) =>
                    ScaleTransition(scale: animation, child: child),
                child: showSend
                    ? GestureDetector(
                        key: const ValueKey('send'),
                        onTap: _isThinking ? null : _sendMessage,
                        child: AnimatedContainer(
                          duration: const Duration(milliseconds: 150),
                          width: 40,
                          height: 40,
                          decoration: BoxDecoration(
                              color: _isThinking
                                  ? t.surfaceHigh
                                  : t.sendBtnBg,
                              borderRadius: BorderRadius.circular(10)),
                          child: Icon(Icons.arrow_upward_rounded,
                              color: _isThinking
                                  ? t.textHint
                                  : t.sendBtnIcon,
                              size: 18),
                        ),
                      )
                    : GestureDetector(
                        key: const ValueKey('mic'),
                        onTap: _toggleListening,
                        child: AnimatedBuilder(
                          animation: _micPulse,
                          builder: (_, _) => Transform.scale(
                            scale: _isListening
                                ? 1.0 + (_micPulse.value * 0.12)
                                : 1.0,
                            child: Container(
                              width: 40,
                              height: 40,
                              decoration: BoxDecoration(
                                color: _isListening
                                    ? t.textPrimary
                                    : t.surfaceHigh,
                                borderRadius: BorderRadius.circular(10),
                                border: Border.all(
                                    color: _isListening
                                        ? t.textPrimary
                                        : t.border,
                                    width: 0.5),
                              ),
                              child: Icon(
                                  _isListening
                                      ? Icons.mic
                                      : Icons.mic_none_outlined,
                                  color: _isListening
                                      ? t.sendBtnIcon
                                      : t.textSecond,
                                  size: 18),
                            ),
                          ),
                        ),
                      ),
              ),
            ],
          ),
        ),
      ],
    );
  }

  @override
  Widget build(BuildContext context) {
    return ValueListenableBuilder<bool>(
      valueListenable: isDarkMode,
      builder: (context, isDark, _) {
        final t = AppTheme(isDark);
        return Scaffold(
          key: _scaffoldKey,
          backgroundColor: t.bg,
          drawer: _buildDrawer(t),
          appBar: AppBar(
            backgroundColor: t.bg,
            elevation: 0,
            centerTitle: true,
            systemOverlayStyle:
                isDark ? SystemUiOverlayStyle.light : SystemUiOverlayStyle.dark,
            leading: IconButton(
              icon: Icon(Icons.menu, color: t.textSecond, size: 20),
              onPressed: () {
                HapticFeedback.lightImpact();
                _scaffoldKey.currentState?.openDrawer();
              },
            ),
            title: VeilIcon(size: 32, isDark: isDark),
            actions: [
              if (!_isLoading)
                PopupMenuButton<String>(
                  icon: Icon(Icons.more_horiz, color: t.textSecond, size: 20),
                  color: t.surface,
                  shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(12),
                      side: BorderSide(color: t.border, width: 0.5)),
                  onSelected: (value) {
                    if (value == 'save') _saveCurrentChat();
                    if (value == 'new') _newChat();
                  },
                  itemBuilder: (_) => [
                    PopupMenuItem(
                        value: 'save',
                        child: Row(children: [
                          Icon(Icons.bookmark_outline,
                              color: t.textSecond, size: 16),
                          const SizedBox(width: 10),
                          Text('Save chat',
                              style: TextStyle(
                                  color: t.textPrimary, fontSize: 14))
                        ])),
                    PopupMenuItem(
                        value: 'new',
                        child: Row(children: [
                          Icon(Icons.add_comment_outlined,
                              color: t.textSecond, size: 16),
                          const SizedBox(width: 10),
                          Text('New chat',
                              style: TextStyle(
                                  color: t.textPrimary, fontSize: 14))
                        ])),
                  ],
                ),
              const SizedBox(width: 4),
            ],
            bottom: PreferredSize(
                preferredSize: const Size.fromHeight(0.5),
                child: Container(height: 0.5, color: t.border)),
          ),
          body: _isLoading
              ? _buildLoadingScreen(t)
              : Column(children: [
                  Expanded(child: _buildMessageList(t)),
                  _buildInputArea(t)
                ]),
        );
      },
    );
  }
}

// ─── Veil Icon ────────────────────────────────────────────────────────────────

class VeilIcon extends StatelessWidget {
  final double size;
  final bool isDark;
  const VeilIcon({super.key, this.size = 24, required this.isDark});

  @override
  Widget build(BuildContext context) {
    return CustomPaint(
        size: Size(size, size), painter: _VeilIconPainter(isDark: isDark));
  }
}

class _VeilIconPainter extends CustomPainter {
  final bool isDark;
  const _VeilIconPainter({required this.isDark});

  @override
  void paint(Canvas canvas, Size size) {
    final w = size.width;
    final h = size.height;
    final colors = isDark
        ? [
            const Color(0xFF141414),
            const Color(0xFF383838),
            const Color(0xFF707070),
            const Color(0xFFB0B0B0),
            const Color(0xFFEFEFEF)
          ]
        : [
            const Color(0xFFDDDDDD),
            const Color(0xFFBBBBBB),
            const Color(0xFF888888),
            const Color(0xFF444444),
            const Color(0xFF111111)
          ];
    final widths = [1.0, 1.4, 1.8, 2.2, 2.6];
    final offsets = [-0.18, -0.10, -0.02, 0.06, 0.14];
    for (int i = 0; i < 5; i++) {
      final paint = Paint()
        ..color = colors[i]
        ..strokeWidth = widths[i]
        ..strokeCap = StrokeCap.round
        ..style = PaintingStyle.stroke;
      final cx = w * (0.5 + offsets[i]);
      final path = Path()
        ..moveTo(cx - w * 0.38, h * 0.08)
        ..lineTo(cx, h * 0.92)
        ..lineTo(cx + w * 0.38, h * 0.08);
      canvas.drawPath(path, paint);
    }
  }

  @override
  bool shouldRepaint(_VeilIconPainter old) => old.isDark != isDark;
}