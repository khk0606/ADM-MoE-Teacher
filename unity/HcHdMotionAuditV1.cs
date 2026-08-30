#if UNITY_EDITOR
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

/// <summary>
/// Replays all 18 genuine hc_hd Chair-local motions in room_0101.
///
/// This is a candidate audit, not a selector.  It finds the seated frame and
/// placement, checks seat contact, floor penetration, non-target collision and
/// the normal dataset crop contract, and writes transformed immutable motions
/// plus a JSON report.  Selection happens in a separate Python gate.
/// </summary>
[InitializeOnLoad]
public static class HcHdMotionAuditV1
{
    private const string ScenePath =
        "Assets/Room_RelationalAffordance_0101.unity";
    private const string CandidateRoot =
        "Assets/RelationalAffordance/Motions/hc_hd_candidates_v1";
    private const string OutputRoot =
        "Assets/RelationalAffordance/Motions/hc_hd_audit_v1/room_0101/chair_06";
    private const string ReportRelative =
        "Exports/history_affordance_relational_v1/scenes/room_0101/" +
        "hc_hd_candidate_audit_v1.json";
    private const string RequestRelative = "Library/HcHdMotionAuditV1.request";
    private const string ResultRelative = "Library/HcHdMotionAuditV1.result.txt";
    private const int CollisionFrameStride = 10;
    private const int SkinVertexStride = 20;
    private const float FloorTolerance = 0.04f;
    private static readonly HashSet<string> SelectedTrainSix = new HashSet<string>
    {
        "hc_hd_b_2", "hc_hd_b_3", "hc_hd_b_4",
        "hc_hd_b_5", "hc_hd_b_6", "hc_hd_f_3",
    };
    private static string ActiveSceneId = "room_0101";
    private static string ActiveOutputRoot = OutputRoot;
    private static string ActiveTargetInstanceId = "chair_06";

    [Serializable]
    private sealed class CandidateManifest
    {
        public string schema;
        public string status;
        public int candidate_count;
        public int blend_frames;
        public List<CandidateSource> candidates;
    }

    [Serializable]
    private sealed class CandidateSource
    {
        public string candidate_id;
        public string source_group_id;
        public int go_raw_frames;
        public bool go_dummy_removed;
        public string output_file;
        public string output_sha256;
        public int output_frames;
        public float join_root_gap_m_before_crossfade;
        public float join_root_rotation_deg_before_crossfade;
    }

    [Serializable]
    private sealed class CollisionAudit
    {
        public bool passed;
        public int sampled_frames;
        public int collision_frame_count;
        public int penetrating_vertex_samples;
        public List<int> collision_frames = new List<int>();
        public List<string> collision_objects = new List<string>();
    }

    [Serializable]
    private sealed class FloorAudit
    {
        public bool passed;
        public float ground_surface_y;
        public float maximum_penetration_m;
        public float maximum_clearance_m;
        public float initial_clearance_m;
        public int penetrating_frame_count;
        public List<int> penetrating_frames = new List<int>();
        public int floating_frame_count;
        public List<int> floating_frames = new List<int>();
    }

    [Serializable]
    private sealed class CandidateAudit
    {
        public string candidate_id;
        public string source_group_id;
        public string source_motion_asset;
        public string source_motion_sha256;
        public string audited_motion_asset;
        public string audited_motion_sha256;
        public int total_frames;
        public int seated_frame;
        public int frame_start_inclusive;
        public int frame_end_inclusive;
        public int cmdm_frames_at_20fps;
        public string placement_candidate;
        public float vertical_contact_shift_m;
        public float target_contact_gap_m;
        public bool target_contact_passed;
        public bool dataset_contract_passed;
        public string dataset_contract_report;
        public float[] final_pelvis_world_xyz;
        public float[] final_facing_world_xz;
        public CollisionAudit collision;
        public FloorAudit floor;
        public float join_root_gap_m;
        public float join_root_rotation_deg;
        public float quality_score;
        public bool eligible_for_source_split;
        public List<string> failed_checks = new List<string>();
    }

    [Serializable]
    private sealed class AuditReport
    {
        public string schema;
        public string status;
        public string authorization;
        public string scene_id;
        public string unity_scene;
        public string target_instance_id;
        public string purpose_instance_id;
        public string text;
        public string candidate_manifest_asset;
        public string candidate_manifest_sha256;
        public int candidate_count;
        public int evaluated_count;
        public int eligible_count;
        public int source_group_count;
        public bool purpose_relation_used_as_forward_input;
        public string created_utc;
        public List<CandidateAudit> candidates = new List<CandidateAudit>();
        public List<string> failed_checks = new List<string>();
    }

    private sealed class MotionRows
    {
        public float[][] rows;
    }

    private sealed class Placement
    {
        public string name;
        public Vector3 pelvisWorld;
    }

    private sealed class Trial
    {
        public MotionRows rows;
        public string placement;
        public int seatedFrame;
        public float verticalShift;
        public float contactGap;
        public bool contactPassed;
        public CollisionAudit collision;
        public FloorAudit floor;
        public long score;
    }

    static HcHdMotionAuditV1()
    {
        EditorApplication.delayCall += RunPendingRequest;
    }

    [MenuItem("Tools/Relational Affordance/HC-HD/1. Audit 18 Candidates")]
    public static void RunFromMenu()
    {
        RunAudit();
    }

    [MenuItem("Tools/Relational Affordance/HC-HD/2. Bind Train Six To Room 0102")]
    public static void RunRoom0102FromMenu()
    {
        RunAuditForScene(
            "room_0102",
            "Assets/Room_RelationalAffordance_0102.unity",
            "Chair_06",
            "Desk_01",
            "Assets/RelationalAffordance/Motions/hc_hd_train_v1/room_0102/chair_06",
            "Exports/history_affordance_relational_v1/scenes/room_0102/" +
                "hc_hd_train_binding_v1.json",
            true
        );
    }

    [MenuItem("Tools/Relational Affordance/HC-HD/3. Audit Independent HCW-HDW In Room 0201")]
    public static void RunRoom0201DevelopmentFromMenu()
    {
        RunRoom0201Development();
    }

    public static void RunBatch()
    {
        try
        {
            AuditReport report = RunAudit();
            Debug.Log("[HC_HD_BATCH_COMPLETE] " + report.status);
            EditorApplication.Exit(report.status == "AUDIT_COMPLETE" ? 0 : 1);
        }
        catch (Exception exception)
        {
            Debug.LogException(exception);
            EditorApplication.Exit(1);
        }
    }

    public static void RunRoom0102Batch()
    {
        try
        {
            AuditReport report = RunAuditForScene(
                "room_0102",
                "Assets/Room_RelationalAffordance_0102.unity",
                "Chair_06",
                "Desk_01",
                "Assets/RelationalAffordance/Motions/hc_hd_train_v1/room_0102/chair_06",
                "Exports/history_affordance_relational_v1/scenes/room_0102/" +
                    "hc_hd_train_binding_v1.json",
                true
            );
            Debug.Log("[HC_HD_ROOM0102_BATCH_COMPLETE] " + report.status);
            EditorApplication.Exit(report.status == "TRAIN_BINDING_COMPLETE" ? 0 : 1);
        }
        catch (Exception exception)
        {
            Debug.LogException(exception);
            EditorApplication.Exit(1);
        }
    }

    public static void RunRoom0201DevelopmentBatch()
    {
        try
        {
            AuditReport report = RunRoom0201Development();
            Debug.Log("[HCW_HDW_ROOM0201_BATCH_COMPLETE] " + report.status);
            EditorApplication.Exit(report.status == "AUDIT_COMPLETE" ? 0 : 1);
        }
        catch (Exception exception)
        {
            Debug.LogException(exception);
            EditorApplication.Exit(1);
        }
    }

    private static AuditReport RunRoom0201Development()
    {
        return RunAuditForScene(
            "room_0201",
            "Assets/Room_RelationalAffordance_0201.unity",
            "Chair_02",
            "Desk_01",
            "Assets/RelationalAffordance/Motions/hcw_hdw_development_audit_v1/" +
                "room_0201/chair_02",
            "Exports/history_affordance_relational_v1/scenes/room_0201/" +
                "hcw_hdw_development_candidate_audit_v1.json",
            false,
            "Assets/RelationalAffordance/Motions/hcw_hdw_development_candidates_v1",
            "relational_affordance_hcw_hdw_candidates_v1",
            "relational_affordance_hcw_hdw_development_unity_audit_v1",
            6
        );
    }

    private static void RunPendingRequest()
    {
        string request = Path.Combine(ProjectRoot(), RequestRelative);
        if (!File.Exists(request)) return;
        string result = Path.Combine(ProjectRoot(), ResultRelative);
        try
        {
            File.Delete(request);
            if (File.Exists(result)) File.Delete(result);
            AuditReport report = RunAudit();
            File.WriteAllText(
                result,
                "status=" + report.status + "\n" +
                "evaluated=" + report.evaluated_count + "/" +
                    report.candidate_count + "\n" +
                "eligible=" + report.eligible_count + "\n" +
                "report=" + Path.Combine(ProjectRoot(), ReportRelative) + "\n"
            );
        }
        catch (Exception exception)
        {
            Debug.LogException(exception);
            File.WriteAllText(result, "status=ERROR\n" + exception + "\n");
        }
    }

    private static AuditReport RunAudit()
    {
        return RunAuditForScene(
            "room_0101", ScenePath, "Chair_06", "Desk_01",
            OutputRoot, ReportRelative, false
        );
    }

    private static AuditReport RunAuditForScene(
        string sceneId,
        string scenePath,
        string targetName,
        string purposeName,
        string outputRoot,
        string reportRelative,
        bool trainBindingMode,
        string candidateRoot = CandidateRoot,
        string candidateManifestSchema = "relational_affordance_hc_hd_candidates_v1",
        string auditReportSchema = null,
        int minimumEligible = 12
    )
    {
        ActiveSceneId = sceneId;
        ActiveOutputRoot = outputRoot;
        ActiveTargetInstanceId = targetName.ToLowerInvariant();
        EditorSceneManager.OpenScene(scenePath, OpenSceneMode.Single);
        Transform coordinate = RequireTransform("ChairCoordinate");
        Transform sceneGeometry = RequireTransform("SceneGeometry");
        Transform target = RequireTransform(targetName);
        Transform desk = RequireTransform(purposeName);
        HistoryDatasetSampleCollector collector;
        HistoryDatasetMotionPreview preview;
        ResolvePreviewRig(out collector, out preview);
        bool previousLogging = preview.logAppliedFrames;
        preview.logAppliedFrames = false;

        string manifestPath = Path.Combine(ProjectRoot(), candidateRoot, "manifest.json");
        if (!File.Exists(manifestPath))
            throw new FileNotFoundException("hc_hd manifest is missing", manifestPath);
        CandidateManifest manifest = JsonUtility.FromJson<CandidateManifest>(
            File.ReadAllText(manifestPath)
        );
        if (manifest == null || manifest.schema != candidateManifestSchema ||
            manifest.status != "CANDIDATE_BUILD_PASS" || manifest.candidate_count != 18 ||
            manifest.candidates == null || manifest.candidates.Count != 18)
        {
            throw new InvalidDataException("hc_hd candidate manifest contract changed");
        }

        AuditReport report = new AuditReport
        {
            schema = trainBindingMode
                ? "relational_affordance_hc_hd_train_binding_v1"
                : (auditReportSchema ?? "relational_affordance_hc_hd_unity_audit_v1"),
            status = "AUDIT_INCOMPLETE",
            authorization = "selection_requires_separate_source_split_gate",
            scene_id = sceneId,
            unity_scene = scenePath,
            target_instance_id = targetName.ToLowerInvariant(),
            purpose_instance_id = purposeName.ToLowerInvariant(),
            text = "Sit anywhere to write.",
            candidate_manifest_asset = candidateRoot + "/manifest.json",
            candidate_manifest_sha256 = Sha256(manifestPath),
            candidate_count = manifest.candidate_count,
            evaluated_count = 0,
            eligible_count = 0,
            source_group_count = manifest.candidates.Select(x => x.source_group_id).Distinct().Count(),
            purpose_relation_used_as_forward_input = false,
            created_utc = DateTime.UtcNow.ToString("o", CultureInfo.InvariantCulture),
        };

        try
        {
            Directory.CreateDirectory(Path.Combine(ProjectRoot(), outputRoot));
            for (int index = 0; index < manifest.candidates.Count; index++)
            {
                CandidateSource source = manifest.candidates[index];
                CandidateAudit audit = AuditOne(
                    source, manifest.blend_frames, candidateRoot, coordinate, sceneGeometry,
                    target, desk, collector, preview
                );
                report.candidates.Add(audit);
                report.evaluated_count++;
                if (audit.eligible_for_source_split) report.eligible_count++;
                Debug.Log(
                    "[HC_HD " + (index + 1).ToString("D2") + "/18] " +
                    audit.candidate_id + " eligible=" +
                    audit.eligible_for_source_split + " frame=" +
                    audit.seated_frame + " contact=" +
                    audit.target_contact_gap_m.ToString("F5", CultureInfo.InvariantCulture) +
                    " collision=" + audit.collision.collision_frame_count +
                    " floor=" + audit.floor.maximum_penetration_m.ToString(
                        "F5", CultureInfo.InvariantCulture)
                );
            }
        }
        finally
        {
            preview.logAppliedFrames = previousLogging;
            EditorUtility.SetDirty(preview);
        }

        if (report.evaluated_count != 18)
            report.failed_checks.Add("all_18_candidates_evaluated");
        if (report.source_group_count != 18)
            report.failed_checks.Add("all_18_source_groups_unique");
        if (trainBindingMode)
        {
            HashSet<string> eligibleIds = new HashSet<string>(
                report.candidates
                    .Where(x => x.eligible_for_source_split)
                    .Select(x => x.candidate_id)
            );
            if (!SelectedTrainSix.IsSubsetOf(eligibleIds))
                report.failed_checks.Add("selected_train_six_physically_valid");
            report.status = report.failed_checks.Count == 0
                ? "TRAIN_BINDING_COMPLETE" : "TRAIN_BINDING_FAIL";
        }
        else
        {
            if (report.eligible_count < minimumEligible)
                report.failed_checks.Add("minimum_physically_eligible_candidates");
            report.status = report.failed_checks.Count == 0
                ? "AUDIT_COMPLETE" : "AUDIT_FAIL";
        }

        string reportPath = Path.Combine(ProjectRoot(), reportRelative);
        Directory.CreateDirectory(Path.GetDirectoryName(reportPath));
        File.WriteAllText(reportPath, JsonUtility.ToJson(report, true) + "\n");
        AssetDatabase.Refresh();
        Debug.Log(
            "[" + report.status + "] hc_hd Unity contact/collision/floor audit\n" +
            "[OK] evaluated: " + report.evaluated_count + "/18\n" +
            "[OK] eligible: " + report.eligible_count + "/18\n" +
            "[OK] train binding mode: " + trainBindingMode + "\n" +
            "[OK] report: " + reportPath
        );
        return report;
    }

    private static CandidateAudit AuditOne(
        CandidateSource source,
        int blendFrames,
        string candidateRoot,
        Transform coordinate,
        Transform sceneGeometry,
        Transform target,
        Transform desk,
        HistoryDatasetSampleCollector collector,
        HistoryDatasetMotionPreview preview
    )
    {
        string sourcePath = Path.Combine(ProjectRoot(), candidateRoot, source.output_file);
        if (Sha256(sourcePath) != source.output_sha256)
            throw new InvalidDataException("candidate hash changed: " + source.candidate_id);
        MotionRows original = ReadMotion(sourcePath);
        if (original.rows.Length != source.output_frames)
            throw new InvalidDataException("candidate frame count changed: " + source.candidate_id);
        int goCleanFrames = source.go_raw_frames - (source.go_dummy_removed ? 1 : 0);
        int nominalSeatFrame = Mathf.Clamp(goCleanFrames - blendFrames, 0, original.rows.Length - 1);
        Vector3 desiredFacing = Flatten(desk.position - target.position);

        // First locate the best seated frame using the center of the target.
        List<Placement> placements = BuildChairPlacements(target, original, coordinate, nominalSeatFrame);
        Placement center = placements.First(x => x.name == "seat_x0.50_z0.50");
        int scanStart = Mathf.Max(0, nominalSeatFrame - 40);
        int scanEnd = Mathf.Min(original.rows.Length - 1, nominalSeatFrame + 20);
        List<int> seatFrames = new List<int>();
        for (int frame = scanStart; frame <= scanEnd; frame += 5) seatFrames.Add(frame);
        if (!seatFrames.Contains(nominalSeatFrame)) seatFrames.Add(nominalSeatFrame);
        Trial bestFrame = null;
        foreach (int frame in seatFrames.OrderBy(x => x))
        {
            Trial trial = MakeTrial(
                original, coordinate, sceneGeometry, target, desiredFacing,
                collector, preview, center, frame, false
            );
            long score =
                (trial.contactPassed ? 0L : 1000000000000L) +
                (long)(Mathf.Abs(trial.contactGap) * 100000000.0f) +
                Math.Abs(frame - nominalSeatFrame);
            trial.score = score;
            if (bestFrame == null || trial.score < bestFrame.score) bestFrame = trial;
        }
        if (bestFrame == null)
            throw new InvalidOperationException("no seated frame evaluated: " + source.candidate_id);

        // Then evaluate every seat placement with full route checks.
        Trial best = null;
        foreach (Placement placement in placements)
        {
            Trial trial = MakeTrial(
                original, coordinate, sceneGeometry, target, desiredFacing,
                collector, preview, placement, bestFrame.seatedFrame, true
            );
            trial.score =
                (trial.contactPassed ? 0L : 1000000000000L) +
                (trial.floor.passed ? 0L : 100000000000L) +
                trial.collision.collision_frame_count * 100000000L +
                trial.collision.penetrating_vertex_samples * 1000L +
                (long)(Mathf.Abs(trial.contactGap) * 1000000.0f);
            if (best == null || trial.score < best.score) best = trial;
        }
        if (best == null)
            throw new InvalidOperationException("no placement evaluated: " + source.candidate_id);

        int frameEnd = best.seatedFrame;
        int frameStart = Mathf.Max(0, frameEnd - 292);
        string outputName = source.candidate_id + "_" +
            ActiveSceneId.Replace("_", "") + "_" +
            ActiveTargetInstanceId.Replace("_", "") + ".txt";
        string outputPath = Path.Combine(ProjectRoot(), ActiveOutputRoot, outputName);
        WriteMotion(outputPath, best.rows);
        Configure(
            collector, preview, coordinate, outputPath, frameStart, frameEnd,
            source.candidate_id, target
        );
        preview.ApplyFrame(frameStart);
        collector.CaptureCurrentMotionRootAsStart();
        preview.ApplyEndFrameAndMeasureSeatContact();
        string contractReport;
        bool datasetPassed = collector.ValidateCurrentSample(out contractReport);
        Vector3 finalPelvis = coordinate.TransformPoint(Root(best.rows.rows[frameEnd]));
        Vector3 finalFacing = Flatten(
            coordinate.rotation * RootRotation(best.rows.rows[frameEnd]) * Vector3.back
        );

        CandidateAudit audit = new CandidateAudit
        {
            candidate_id = source.candidate_id,
            source_group_id = source.source_group_id,
            source_motion_asset = candidateRoot + "/" + source.output_file,
            source_motion_sha256 = source.output_sha256,
            audited_motion_asset = ActiveOutputRoot + "/" + outputName,
            audited_motion_sha256 = Sha256(outputPath),
            total_frames = best.rows.rows.Length,
            seated_frame = frameEnd,
            frame_start_inclusive = frameStart,
            frame_end_inclusive = frameEnd,
            cmdm_frames_at_20fps = ExpectedResampledFrameCount(
                frameEnd - frameStart + 1, 30, 20
            ),
            placement_candidate = best.placement,
            vertical_contact_shift_m = best.verticalShift,
            target_contact_gap_m = best.contactGap,
            target_contact_passed = best.contactPassed,
            dataset_contract_passed = datasetPassed,
            dataset_contract_report = contractReport,
            final_pelvis_world_xyz = new float[]
                { finalPelvis.x, finalPelvis.y, finalPelvis.z },
            final_facing_world_xz = new float[] { finalFacing.x, finalFacing.z },
            collision = best.collision,
            floor = best.floor,
            join_root_gap_m = source.join_root_gap_m_before_crossfade,
            join_root_rotation_deg = source.join_root_rotation_deg_before_crossfade,
        };
        if (!audit.target_contact_passed) audit.failed_checks.Add("target_contact_passes");
        if (!audit.collision.passed) audit.failed_checks.Add("non_target_collision_passes");
        if (!audit.floor.passed) audit.failed_checks.Add("floor_penetration_passes");
        if (!audit.dataset_contract_passed) audit.failed_checks.Add("dataset_crop_contract_passes");
        if (audit.cmdm_frames_at_20fps > 196) audit.failed_checks.Add("cmdm_frames_at_most_196");
        audit.eligible_for_source_split = audit.failed_checks.Count == 0;
        audit.quality_score =
            Mathf.Abs(audit.target_contact_gap_m) * 1000.0f +
            audit.floor.maximum_penetration_m * 1000.0f +
            audit.collision.collision_frame_count * 100.0f +
            audit.collision.penetrating_vertex_samples +
            audit.join_root_gap_m * 100.0f +
            audit.join_root_rotation_deg * 0.01f;
        return audit;
    }

    private static Trial MakeTrial(
        MotionRows original,
        Transform coordinate,
        Transform sceneGeometry,
        Transform target,
        Vector3 desiredFacing,
        HistoryDatasetSampleCollector collector,
        HistoryDatasetMotionPreview preview,
        Placement placement,
        int seatedFrame,
        bool fullAudit
    )
    {
        Vector3 sourceFinalWorld = coordinate.TransformPoint(
            Root(original.rows[seatedFrame])
        );
        Quaternion sourceFinalRotation = coordinate.rotation *
            RootRotation(original.rows[seatedFrame]);
        Quaternion yawDelta = YawDelta(sourceFinalRotation, desiredFacing);
        MotionRows transformed = TransformRigid(
            original, coordinate, sourceFinalWorld, placement.pelvisWorld, yawDelta
        );
        string temporary = Path.Combine(
            ProjectRoot(), "Library/HcHdMotionAuditV1", placement.name + ".txt"
        );
        WriteMotion(temporary, transformed);
        int frameStart = Mathf.Max(0, seatedFrame - 292);
        Configure(
            collector, preview, coordinate, temporary, frameStart, seatedFrame,
            "hc_hd_trial", target
        );
        float shift = 0.0f;
        if (fullAudit)
        {
            FloorAudit initialFloor = AuditFloor(
                preview, sceneGeometry, frameStart, seatedFrame
            );
            float baseLift = Mathf.Max(
                0.0f,
                initialFloor.maximum_penetration_m - FloorTolerance + 0.002f
            );
            if (baseLift > 0.0f && baseLift <= 0.08f)
            {
                transformed = ShiftWorldY(transformed, coordinate, baseLift);
                shift += baseLift;
                WriteMotion(temporary, transformed);
            }
        }
        preview.ApplyFrame(seatedFrame);
        MeasureSeatContactQuietly(preview);
        for (int correctionIndex = 0; correctionIndex < 6; correctionIndex++)
        {
            if (preview.lastSeatContactPassed) break;
            if (float.IsNaN(preview.lastSeatGapMeters) ||
                float.IsInfinity(preview.lastSeatGapMeters)) break;
            float remaining = Mathf.Max(0.0f, 0.35f - Mathf.Abs(shift));
            float correction = Mathf.Clamp(
                -preview.lastSeatGapMeters, -remaining, remaining
            );
            if (Mathf.Abs(correction) < 0.001f) break;
            shift += correction;
            transformed = ShiftWorldYAroundSeat(
                transformed, coordinate, correction, seatedFrame, 45
            );
            WriteMotion(temporary, transformed);
            preview.ApplyFrame(seatedFrame);
            MeasureSeatContactQuietly(preview);
        }
        CollisionAudit collision = fullAudit
            ? AuditCollision(preview, sceneGeometry, target, frameStart, seatedFrame)
            : new CollisionAudit { passed = true };
        FloorAudit floor = fullAudit
            ? AuditFloor(preview, sceneGeometry, frameStart, seatedFrame)
            : new FloorAudit { passed = true };
        return new Trial
        {
            rows = transformed,
            placement = placement.name,
            seatedFrame = seatedFrame,
            verticalShift = shift,
            contactGap = preview.lastSeatGapMeters,
            contactPassed = preview.lastSeatContactPassed,
            collision = collision,
            floor = floor,
        };
    }

    private static List<Placement> BuildChairPlacements(
        Transform target,
        MotionRows original,
        Transform coordinate,
        int nominalSeatFrame
    )
    {
        Renderer[] renderers = target.GetComponentsInChildren<Renderer>(true);
        if (renderers.Length == 0)
            throw new InvalidOperationException("Chair_06 has no Renderer bounds");
        Bounds bounds = renderers[0].bounds;
        for (int index = 1; index < renderers.Length; index++)
            bounds.Encapsulate(renderers[index].bounds);
        float insetX = Mathf.Min(0.12f, bounds.size.x * 0.20f);
        float insetZ = Mathf.Min(0.12f, bounds.size.z * 0.20f);
        float minX = bounds.min.x + insetX;
        float maxX = bounds.max.x - insetX;
        float minZ = bounds.min.z + insetZ;
        float maxZ = bounds.max.z - insetZ;
        float pelvisY = coordinate.TransformPoint(
            Root(original.rows[nominalSeatFrame])
        ).y;
        float[] fractions = new float[] { 0.50f, 0.35f, 0.65f };
        List<Placement> result = new List<Placement>();
        foreach (float xFraction in fractions)
        {
            foreach (float zFraction in fractions)
            {
                result.Add(new Placement
                {
                    name = "seat_x" + xFraction.ToString("F2", CultureInfo.InvariantCulture) +
                        "_z" + zFraction.ToString("F2", CultureInfo.InvariantCulture),
                    pelvisWorld = new Vector3(
                        Mathf.Lerp(minX, maxX, xFraction), pelvisY,
                        Mathf.Lerp(minZ, maxZ, zFraction)
                    ),
                });
            }
        }
        return result;
    }

    private static CollisionAudit AuditCollision(
        HistoryDatasetMotionPreview preview,
        Transform sceneGeometry,
        Transform targetRoot,
        int frameStart,
        int frameEnd
    )
    {
        Collider[] colliders = sceneGeometry.GetComponentsInChildren<Collider>(true)
            .Where(x => x.enabled && !x.isTrigger && !IsGround(x.transform))
            .ToArray();
        SkinnedMeshRenderer[] skins =
            preview.actor.GetComponentsInChildren<SkinnedMeshRenderer>(true);
        CollisionAudit result = new CollisionAudit();
        HashSet<string> objectNames = new HashSet<string>(StringComparer.Ordinal);
        bool firstSample = true;
        foreach (int frame in SampleFrames(frameStart, frameEnd))
        {
            result.sampled_frames++;
            if (!preview.ApplyFrame(frame))
            {
                result.collision_frame_count++;
                result.collision_frames.Add(frame);
                objectNames.Add("<pose_apply_failed>");
                continue;
            }
            Physics.SyncTransforms();
            int framePenetrations = 0;
            foreach (SkinnedMeshRenderer skin in skins)
            {
                Mesh baked = new Mesh();
                try
                {
                    skin.BakeMesh(baked);
                    Vector3[] vertices = baked.vertices;
                    Matrix4x4 localToWorld = skin.transform.localToWorldMatrix;
                    Bounds skinBounds = skin.bounds;
                    foreach (Collider collider in colliders)
                    {
                        if (!skinBounds.Intersects(collider.bounds)) continue;
                        if (collider.transform == targetRoot ||
                            collider.transform.IsChildOf(targetRoot)) continue;
                        for (int vertex = 0; vertex < vertices.Length;
                             vertex += SkinVertexStride)
                        {
                            Vector3 world = localToWorld.MultiplyPoint3x4(vertices[vertex]);
                            if (!collider.bounds.Contains(world)) continue;
                            Vector3 closest = collider.ClosestPoint(world);
                            if ((closest - world).sqrMagnitude <= 1.0e-10f)
                            {
                                framePenetrations++;
                                objectNames.Add(CollisionName(collider.transform));
                            }
                        }
                    }
                }
                finally
                {
                    UnityEngine.Object.DestroyImmediate(baked);
                }
            }
            if (framePenetrations > 0)
            {
                result.collision_frame_count++;
                result.penetrating_vertex_samples += framePenetrations;
                result.collision_frames.Add(frame);
            }
        }
        result.collision_objects = objectNames.OrderBy(x => x).ToList();
        result.passed = result.collision_frame_count == 0;
        return result;
    }

    private static FloorAudit AuditFloor(
        HistoryDatasetMotionPreview preview,
        Transform sceneGeometry,
        int frameStart,
        int frameEnd
    )
    {
        Transform ground = RequireTransform("Ground");
        Renderer[] groundRenderers = ground.GetComponentsInChildren<Renderer>(true);
        Collider[] groundColliders = ground.GetComponentsInChildren<Collider>(true)
            .Where(x => x.enabled).ToArray();
        if (groundRenderers.Length == 0 && groundColliders.Length == 0)
            throw new InvalidOperationException("Ground has no bounds provider");
        float groundY = groundColliders.Length > 0
            ? groundColliders.Max(x => x.bounds.max.y)
            : groundRenderers.Max(x => x.bounds.max.y);
        FloorAudit result = new FloorAudit { ground_surface_y = groundY };
        SkinnedMeshRenderer[] skins =
            preview.actor.GetComponentsInChildren<SkinnedMeshRenderer>(true);
        bool firstSample = true;
        foreach (int frame in SampleFrames(frameStart, frameEnd))
        {
            if (!preview.ApplyFrame(frame))
            {
                result.penetrating_frame_count++;
                result.penetrating_frames.Add(frame);
                continue;
            }
            float minimumY = float.PositiveInfinity;
            foreach (SkinnedMeshRenderer skin in skins)
            {
                Mesh baked = new Mesh();
                try
                {
                    skin.BakeMesh(baked);
                    Matrix4x4 localToWorld = skin.transform.localToWorldMatrix;
                    Vector3[] vertices = baked.vertices;
                    for (int vertex = 0; vertex < vertices.Length;
                         vertex += SkinVertexStride)
                    {
                        minimumY = Mathf.Min(
                            minimumY,
                            localToWorld.MultiplyPoint3x4(vertices[vertex]).y
                        );
                    }
                }
                finally
                {
                    UnityEngine.Object.DestroyImmediate(baked);
                }
            }
            float penetration = Mathf.Max(0.0f, groundY - minimumY);
            float clearance = Mathf.Max(0.0f, minimumY - groundY);
            result.maximum_penetration_m = Mathf.Max(
                result.maximum_penetration_m, penetration
            );
            result.maximum_clearance_m = Mathf.Max(
                result.maximum_clearance_m, clearance
            );
            if (firstSample)
            {
                result.initial_clearance_m = clearance;
                firstSample = false;
            }
            if (penetration > FloorTolerance)
            {
                result.penetrating_frame_count++;
                result.penetrating_frames.Add(frame);
            }
            if (clearance > 0.08f)
            {
                result.floating_frame_count++;
                result.floating_frames.Add(frame);
            }
        }
        // A seated pose may legitimately leave both feet above the ground.
        // Clearance is recorded for diagnosis, but the hard floor gate is
        // penetration only.  The seat-contact gate separately checks support.
        result.passed = result.penetrating_frame_count == 0;
        return result;
    }

    private static void MeasureSeatContactQuietly(
        HistoryDatasetMotionPreview preview
    )
    {
        bool previous = Debug.unityLogger.logEnabled;
        try
        {
            Debug.unityLogger.logEnabled = false;
            preview.MeasureCurrentSeatContact();
        }
        finally
        {
            Debug.unityLogger.logEnabled = previous;
        }
    }

    private static List<int> SampleFrames(int frameStart, int frameEnd)
    {
        List<int> frames = new List<int>();
        for (int frame = frameStart; frame <= frameEnd; frame += CollisionFrameStride)
            frames.Add(frame);
        if (!frames.Contains(frameEnd)) frames.Add(frameEnd);
        return frames;
    }

    private static void Configure(
        HistoryDatasetSampleCollector collector,
        HistoryDatasetMotionPreview preview,
        Transform coordinate,
        string path,
        int frameStart,
        int frameEnd,
        string candidateId,
        Transform target
    )
    {
        collector.sceneId = ActiveSceneId;
        collector.sampleId = ActiveSceneId + "_write_" + candidateId;
        collector.textPrompt = "Sit anywhere to write.";
        collector.coordinateFrame = coordinate;
        collector.sourceMotionPath = path;
        collector.sourceFps = 30;
        collector.targetFps = 20;
        collector.frameStart = frameStart;
        collector.frameEnd = frameEnd;
        collector.expectedColumns = 103;
        collector.maximumCmdmFrames = 196;
        collector.targetObject = HistoryDatasetTarget.Chair;
        collector.historyDirectionSource = HistoryDirectionSource.InitialTrajectory;
        collector.historyDirectionSearchFrames = 90;
        collector.historyMinimumPlanarDisplacementMeters = 0.20f;
        collector.hasCapturedStart = false;
        preview.collector = collector;
        preview.chairRoot = target;
        preview.lastSeatContactPassed = false;
        preview.lastSeatGapMeters = float.NaN;
    }

    private static MotionRows TransformRigid(
        MotionRows source,
        Transform coordinate,
        Vector3 sourceFinalWorld,
        Vector3 targetFinalWorld,
        Quaternion worldYawDelta
    )
    {
        float[][] output = new float[source.rows.Length][];
        for (int rowIndex = 0; rowIndex < source.rows.Length; rowIndex++)
        {
            float[] row = (float[])source.rows[rowIndex].Clone();
            Vector3 world = coordinate.TransformPoint(Root(row));
            Vector3 transformedWorld = targetFinalWorld +
                worldYawDelta * (world - sourceFinalWorld);
            SetRoot(row, coordinate.InverseTransformPoint(transformedWorld));
            for (int bone = 0; bone < 25; bone++)
            {
                int column = 3 + bone * 4;
                Quaternion local = new Quaternion(
                    row[column], row[column + 1], row[column + 2], row[column + 3]
                ).normalized;
                Quaternion worldRotation = coordinate.rotation * local;
                Quaternion transformed = Quaternion.Inverse(coordinate.rotation) *
                    worldYawDelta * worldRotation;
                transformed = transformed.normalized;
                row[column] = transformed.x;
                row[column + 1] = transformed.y;
                row[column + 2] = transformed.z;
                row[column + 3] = transformed.w;
            }
            output[rowIndex] = row;
        }
        return new MotionRows { rows = output };
    }

    private static MotionRows ShiftWorldY(
        MotionRows source, Transform coordinate, float worldDeltaY
    )
    {
        Vector3 localDelta = coordinate.InverseTransformVector(
            new Vector3(0.0f, worldDeltaY, 0.0f)
        );
        float[][] output = new float[source.rows.Length][];
        for (int rowIndex = 0; rowIndex < source.rows.Length; rowIndex++)
        {
            float[] row = (float[])source.rows[rowIndex].Clone();
            SetRoot(row, Root(row) + localDelta);
            output[rowIndex] = row;
        }
        return new MotionRows { rows = output };
    }

    private static MotionRows ShiftWorldYAroundSeat(
        MotionRows source,
        Transform coordinate,
        float worldDeltaY,
        int seatedFrame,
        int transitionFrames
    )
    {
        float[][] output = new float[source.rows.Length][];
        int start = Mathf.Max(0, seatedFrame - transitionFrames);
        int end = Mathf.Min(source.rows.Length - 1, seatedFrame + transitionFrames);
        for (int rowIndex = 0; rowIndex < source.rows.Length; rowIndex++)
        {
            float factor;
            if (rowIndex <= seatedFrame)
            {
                factor = Mathf.InverseLerp(start, seatedFrame, rowIndex);
            }
            else
            {
                factor = 1.0f - Mathf.InverseLerp(seatedFrame, end, rowIndex);
            }
            Vector3 localDelta = coordinate.InverseTransformVector(
                new Vector3(0.0f, worldDeltaY * Mathf.Clamp01(factor), 0.0f)
            );
            float[] row = (float[])source.rows[rowIndex].Clone();
            SetRoot(row, Root(row) + localDelta);
            output[rowIndex] = row;
        }
        return new MotionRows { rows = output };
    }

    private static Quaternion YawDelta(
        Quaternion sourceFinalRotation, Vector3 desiredFacing
    )
    {
        Vector3 sourceFacing = Flatten(sourceFinalRotation * Vector3.back);
        Vector3 desired = Flatten(desiredFacing);
        if (sourceFacing.sqrMagnitude < 1.0e-8f || desired.sqrMagnitude < 1.0e-8f)
            throw new InvalidOperationException("cannot resolve motion-facing yaw");
        return Quaternion.AngleAxis(
            Vector3.SignedAngle(sourceFacing, desired, Vector3.up), Vector3.up
        );
    }

    private static Vector3 Flatten(Vector3 value)
    {
        value.y = 0.0f;
        return value.sqrMagnitude < 1.0e-8f ? Vector3.zero : value.normalized;
    }

    private static MotionRows ReadMotion(string path)
    {
        List<float[]> rows = new List<float[]>();
        int lineIndex = 0;
        foreach (string line in File.ReadLines(path))
        {
            if (string.IsNullOrWhiteSpace(line)) continue;
            string[] tokens = line.Split(
                new char[] { ' ', '\t' }, StringSplitOptions.RemoveEmptyEntries
            );
            if (tokens.Length != 103)
                throw new InvalidDataException(
                    path + " row " + lineIndex + " has " + tokens.Length +
                    " columns; expected 103"
                );
            float[] row = new float[103];
            for (int column = 0; column < row.Length; column++)
            {
                row[column] = float.Parse(
                    tokens[column], NumberStyles.Float, CultureInfo.InvariantCulture
                );
                if (float.IsNaN(row[column]) || float.IsInfinity(row[column]))
                    throw new InvalidDataException("non-finite motion value: " + path);
            }
            rows.Add(row);
            lineIndex++;
        }
        return new MotionRows { rows = rows.ToArray() };
    }

    private static void WriteMotion(string path, MotionRows motion)
    {
        string text = string.Join(
            "\n",
            motion.rows.Select(row => string.Join(
                " ", row.Select(value =>
                    value.ToString("R", CultureInfo.InvariantCulture)
                ).ToArray()
            )).ToArray()
        ) + "\n";
        Directory.CreateDirectory(Path.GetDirectoryName(path));
        if (File.Exists(path) && File.ReadAllText(path) == text) return;
        File.WriteAllText(path, text);
    }

    private static Vector3 Root(float[] row)
    {
        return new Vector3(row[0], row[1], row[2]);
    }

    private static void SetRoot(float[] row, Vector3 value)
    {
        row[0] = value.x;
        row[1] = value.y;
        row[2] = value.z;
    }

    private static Quaternion RootRotation(float[] row)
    {
        return new Quaternion(row[3], row[4], row[5], row[6]).normalized;
    }

    private static int ExpectedResampledFrameCount(
        int sourceFrameCount, int sourceRate, int targetRate
    )
    {
        double duration = (sourceFrameCount - 1) / (double)sourceRate;
        return (int)Math.Round(duration * targetRate, MidpointRounding.ToEven) + 1;
    }

    private static string CollisionName(Transform transform)
    {
        RelationalAffordanceObject label =
            transform.GetComponentInParent<RelationalAffordanceObject>();
        if (label != null) return label.InstanceId;
        Transform current = transform;
        while (current.parent != null && current.parent.name != "SceneGeometry")
            current = current.parent;
        return current.name;
    }

    private static bool IsGround(Transform transform)
    {
        Transform current = transform;
        while (current != null)
        {
            if (current.name == "Ground") return true;
            current = current.parent;
        }
        return false;
    }

    private static void ResolvePreviewRig(
        out HistoryDatasetSampleCollector collector,
        out HistoryDatasetMotionPreview preview
    )
    {
        Transform capture = RequireTransform("DatasetCapture");
        collector = capture.GetComponent<HistoryDatasetSampleCollector>();
        preview = capture.GetComponent<HistoryDatasetMotionPreview>();
        if (collector == null || preview == null || preview.actor == null)
            throw new InvalidOperationException("DatasetCapture preview rig is incomplete");
    }

    private static Transform RequireTransform(string name)
    {
        Transform transform = HistoryDatasetSampleCollector.FindSceneTransform(name);
        if (transform == null)
            throw new InvalidOperationException("required scene transform missing: " + name);
        return transform;
    }

    private static string Sha256(string path)
    {
        using (SHA256 sha = SHA256.Create())
        using (FileStream stream = File.OpenRead(path))
            return BitConverter.ToString(sha.ComputeHash(stream)).Replace("-", "").ToLowerInvariant();
    }

    private static string ProjectRoot()
    {
        return Directory.GetParent(Application.dataPath).FullName;
    }
}
#endif
