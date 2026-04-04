import json
from pathlib import Path
from typing import Dict, List, Tuple

from create_npu.golden_model import cluster_control_reference, scheduler_reference, top_npu_reference
from create_npu.models import ArchitectureCandidate, GeneratedDesignBundle, RequirementSpec


def emit_seed_rtl(
    spec: RequirementSpec,
    architecture: ArchitectureCandidate,
    output_dir: Path,
    candidate_id: str = "baseline",
    generator_backend: str = "heuristic",
    extra_notes=None,
    extra_supporting_files=None,
) -> GeneratedDesignBundle:
    rtl_dir = output_dir / "rtl"
    tb_dir = output_dir / "tb"
    rtl_dir.mkdir(parents=True, exist_ok=True)
    tb_dir.mkdir(parents=True, exist_ok=True)

    operand_width, note = _resolve_width(spec.numeric_precision)
    acc_width = max(32, operand_width * 4)
    seed_tile_rows, seed_tile_cols = _seed_tile_shape(
        architecture.tile_rows,
        architecture.tile_cols,
    )

    mac_unit_path = rtl_dir / "mac_unit.sv"
    mac_unit_path.write_text(
        _mac_unit_template(operand_width=operand_width, acc_width=acc_width),
        encoding="utf-8",
    )

    processing_element_path = rtl_dir / "processing_element.sv"
    processing_element_path.write_text(
        _processing_element_template(operand_width=operand_width, acc_width=acc_width),
        encoding="utf-8",
    )

    systolic_tile_path = rtl_dir / "systolic_tile.sv"
    systolic_tile_path.write_text(
        _systolic_tile_template(
            operand_width=operand_width,
            acc_width=acc_width,
            seed_rows=seed_tile_rows,
            seed_cols=seed_tile_cols,
        ),
        encoding="utf-8",
    )

    scratchpad_controller_path = rtl_dir / "scratchpad_controller.sv"
    scratchpad_controller_path.write_text(
        _scratchpad_controller_template(
            operand_width=operand_width,
            seed_rows=seed_tile_rows,
            seed_cols=seed_tile_cols,
        ),
        encoding="utf-8",
    )

    dma_engine_path = rtl_dir / "dma_engine.sv"
    dma_engine_path.write_text(
        _dma_engine_template(
            operand_width=operand_width,
            seed_rows=seed_tile_rows,
            seed_cols=seed_tile_cols,
        ),
        encoding="utf-8",
    )

    tile_compute_unit_path = rtl_dir / "tile_compute_unit.sv"
    tile_compute_unit_path.write_text(
        _tile_compute_unit_template(
            operand_width=operand_width,
            acc_width=acc_width,
            seed_rows=seed_tile_rows,
            seed_cols=seed_tile_cols,
        ),
        encoding="utf-8",
    )

    scheduler_path = rtl_dir / "scheduler.sv"
    scheduler_path.write_text(
        _scheduler_template(
            operand_width=operand_width,
            seed_rows=seed_tile_rows,
            seed_cols=seed_tile_cols,
        ),
        encoding="utf-8",
    )

    cluster_control_path = rtl_dir / "cluster_control.sv"
    cluster_control_path.write_text(
        _cluster_control_template(seed_tile_count=_seed_tile_count(architecture.tile_count)),
        encoding="utf-8",
    )

    top_npu_path = rtl_dir / "top_npu.sv"
    top_npu_path.write_text(
        _top_npu_template(
            operand_width=operand_width,
            acc_width=acc_width,
            seed_rows=seed_tile_rows,
            seed_cols=seed_tile_cols,
            seed_tile_count=_seed_tile_count(architecture.tile_count),
        ),
        encoding="utf-8",
    )

    mac_tb_path = tb_dir / "mac_unit_tb.sv"
    mac_tb_path.write_text(
        _mac_unit_tb_template(operand_width=operand_width, acc_width=acc_width),
        encoding="utf-8",
    )

    pe_tb_path = tb_dir / "processing_element_tb.sv"
    pe_tb_path.write_text(
        _processing_element_tb_template(operand_width=operand_width, acc_width=acc_width),
        encoding="utf-8",
    )

    tile_tb_path = tb_dir / "systolic_tile_tb.sv"
    tile_tb_path.write_text(
        _systolic_tile_tb_template(operand_width=operand_width, acc_width=acc_width),
        encoding="utf-8",
    )

    tile_rect_tb_path = tb_dir / "systolic_tile_rect_tb.sv"
    tile_rect_tb_path.write_text(
        _systolic_tile_rect_tb_template(operand_width=operand_width, acc_width=acc_width),
        encoding="utf-8",
    )

    scratchpad_tb_path = tb_dir / "scratchpad_controller_tb.sv"
    scratchpad_tb_path.write_text(
        _scratchpad_controller_tb_template(operand_width=operand_width),
        encoding="utf-8",
    )

    dma_tb_path = tb_dir / "dma_engine_tb.sv"
    dma_tb_path.write_text(
        _dma_engine_tb_template(operand_width=operand_width),
        encoding="utf-8",
    )

    tile_compute_tb_path = tb_dir / "tile_compute_unit_tb.sv"
    tile_compute_tb_path.write_text(
        _tile_compute_unit_tb_template(operand_width=operand_width, acc_width=acc_width),
        encoding="utf-8",
    )

    scheduler_tb_path = tb_dir / "scheduler_tb.sv"
    scheduler_tb_path.write_text(
        _scheduler_tb_template(operand_width=operand_width),
        encoding="utf-8",
    )

    cluster_control_tb_path = tb_dir / "cluster_control_tb.sv"
    cluster_control_tb_path.write_text(
        _cluster_control_tb_template(),
        encoding="utf-8",
    )

    top_npu_tb_path = tb_dir / "top_npu_tb.sv"
    top_npu_tb_path.write_text(
        _top_npu_tb_template(operand_width=operand_width, acc_width=acc_width),
        encoding="utf-8",
    )

    intent_path = output_dir / "design_intent.md"
    intent_path.write_text(
        _design_intent_template(spec=spec, architecture=architecture),
        encoding="utf-8",
    )

    reference_cases_path = output_dir / "verification_vectors.json"
    reference_cases_path.write_text(
        json.dumps(
            _reference_cases(),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    notes = []
    if note:
        notes.append(note)
    if extra_notes:
        notes.extend(extra_notes)

    if seed_tile_rows != architecture.tile_rows or seed_tile_cols != architecture.tile_cols:
        notes.append(
            "Il seed RTL usa una tile di default "
            f"{seed_tile_rows}x{seed_tile_cols}, derivata dalla tile architetturale "
            f"{architecture.tile_rows}x{architecture.tile_cols} e ridotta per mantenere "
            "la verifica locale trattabile."
        )
    else:
        notes.append(
            "Il seed RTL propaga direttamente la tile architetturale "
            f"{seed_tile_rows}x{seed_tile_cols} nei default del cluster."
        )

    notes.append(
        "Il bundle seed include `mac_unit`, `processing_element`, `systolic_tile`, `dma_engine`, `scratchpad_controller`, `tile_compute_unit`, `scheduler`, `cluster_control` e `top_npu`."
    )
    notes.append(
        "Il cluster seed separa ora il control-path nel modulo `cluster_control`, lasciando a `top_npu` il solo wiring del data-path e l'aggregazione dei risultati."
    )

    supporting_files = [str(intent_path), str(reference_cases_path)]
    if extra_supporting_files:
        supporting_files.extend(extra_supporting_files)

    return GeneratedDesignBundle(
        rtl_files=[
            str(mac_unit_path),
            str(processing_element_path),
            str(systolic_tile_path),
            str(dma_engine_path),
            str(scratchpad_controller_path),
            str(tile_compute_unit_path),
            str(scheduler_path),
            str(cluster_control_path),
            str(top_npu_path),
        ],
        testbench_files=[
            str(mac_tb_path),
            str(pe_tb_path),
            str(tile_tb_path),
            str(tile_rect_tb_path),
            str(dma_tb_path),
            str(scratchpad_tb_path),
            str(tile_compute_tb_path),
            str(scheduler_tb_path),
            str(cluster_control_tb_path),
            str(top_npu_tb_path),
        ],
        primary_module="top_npu",
        operand_width_bits=operand_width,
        supporting_files=supporting_files,
        reference_cases_path=str(reference_cases_path),
        candidate_id=candidate_id,
        generator_backend=generator_backend,
        notes=notes,
    )


def _resolve_width(precision: str) -> Tuple[int, str]:
    if precision == "INT8":
        return 8, ""
    if precision == "INT16":
        return 16, ""
    return 16, (
        f"Precisione {precision} non implementata come aritmetica reale nell'MVP: "
        "uso una rappresentazione integer signed a 16 bit come seed RTL."
    )


def _seed_tile_count(tile_count: int) -> int:
    return max(1, min(int(tile_count), 4))


def _seed_tile_shape(tile_rows: int, tile_cols: int, max_edge: int = 4) -> Tuple[int, int]:
    return (
        max(1, min(int(tile_rows), max_edge)),
        max(1, min(int(tile_cols), max_edge)),
    )


def _mac_unit_template(operand_width: int, acc_width: int) -> str:
    return f"""module mac_unit #(
  parameter int DATA_WIDTH = {operand_width},
  parameter int ACC_WIDTH = {acc_width}
) (
  input  logic signed [DATA_WIDTH-1:0] a,
  input  logic signed [DATA_WIDTH-1:0] b,
  input  logic signed [ACC_WIDTH-1:0] acc_in,
  output logic signed [ACC_WIDTH-1:0] acc_out
);
  logic signed [2*DATA_WIDTH-1:0] product;
  logic signed [ACC_WIDTH-1:0] product_ext;

  always_comb begin
    product = a * b;
    product_ext = product;
    acc_out = acc_in + product_ext;
  end
endmodule
"""


def _mac_unit_tb_template(operand_width: int, acc_width: int) -> str:
    return f"""module mac_unit_tb;
  localparam int DATA_WIDTH = {operand_width};
  localparam int ACC_WIDTH = {acc_width};

  logic signed [DATA_WIDTH-1:0] a;
  logic signed [DATA_WIDTH-1:0] b;
  logic signed [ACC_WIDTH-1:0] acc_in;
  logic signed [ACC_WIDTH-1:0] acc_out;

  mac_unit #(
    .DATA_WIDTH(DATA_WIDTH),
    .ACC_WIDTH(ACC_WIDTH)
  ) dut (
    .a(a),
    .b(b),
    .acc_in(acc_in),
    .acc_out(acc_out)
  );

  initial begin
    a = 4;
    b = 3;
    acc_in = 2;
    #1;
    if (acc_out !== 14) begin
      $fatal(1, "Test 1 failed: expected 14, got %0d", acc_out);
    end

    a = -5;
    b = 6;
    acc_in = 10;
    #1;
    if (acc_out !== -20) begin
      $fatal(1, "Test 2 failed: expected -20, got %0d", acc_out);
    end

    $display("mac_unit_tb passed");
    $finish;
  end
endmodule
"""


def _processing_element_template(operand_width: int, acc_width: int) -> str:
    return f"""module processing_element #(
  parameter int DATA_WIDTH = {operand_width},
  parameter int ACC_WIDTH = {acc_width}
) (
  input  logic signed [DATA_WIDTH-1:0] activation_i,
  input  logic signed [DATA_WIDTH-1:0] weight_i,
  input  logic signed [ACC_WIDTH-1:0] psum_i,
  input  logic compute_en,
  input  logic clear_acc,
  output logic signed [ACC_WIDTH-1:0] psum_o,
  output logic valid_o
);
  logic signed [ACC_WIDTH-1:0] mac_result;

  mac_unit #(
    .DATA_WIDTH(DATA_WIDTH),
    .ACC_WIDTH(ACC_WIDTH)
  ) mac_inst (
    .a(activation_i),
    .b(weight_i),
    .acc_in(psum_i),
    .acc_out(mac_result)
  );

  always_comb begin
    if (clear_acc) begin
      psum_o = '0;
      valid_o = 1'b0;
    end else if (compute_en) begin
      psum_o = mac_result;
      valid_o = 1'b1;
    end else begin
      psum_o = psum_i;
      valid_o = 1'b0;
    end
  end
endmodule
"""


def _processing_element_tb_template(operand_width: int, acc_width: int) -> str:
    return f"""module processing_element_tb;
  localparam int DATA_WIDTH = {operand_width};
  localparam int ACC_WIDTH = {acc_width};

  logic signed [DATA_WIDTH-1:0] activation_i;
  logic signed [DATA_WIDTH-1:0] weight_i;
  logic signed [ACC_WIDTH-1:0] psum_i;
  logic compute_en;
  logic clear_acc;
  logic signed [ACC_WIDTH-1:0] psum_o;
  logic valid_o;

  processing_element #(
    .DATA_WIDTH(DATA_WIDTH),
    .ACC_WIDTH(ACC_WIDTH)
  ) dut (
    .activation_i(activation_i),
    .weight_i(weight_i),
    .psum_i(psum_i),
    .compute_en(compute_en),
    .clear_acc(clear_acc),
    .psum_o(psum_o),
    .valid_o(valid_o)
  );

  initial begin
    activation_i = 3;
    weight_i = 4;
    psum_i = 5;
    compute_en = 1'b1;
    clear_acc = 1'b0;
    #1;
    if (psum_o !== 17 || valid_o !== 1'b1) begin
      $fatal(1, "PE test 1 failed: expected (17, 1), got (%0d, %0d)", psum_o, valid_o);
    end

    activation_i = -2;
    weight_i = 5;
    psum_i = 11;
    compute_en = 1'b0;
    clear_acc = 1'b0;
    #1;
    if (psum_o !== 11 || valid_o !== 1'b0) begin
      $fatal(1, "PE test 2 failed: expected (11, 0), got (%0d, %0d)", psum_o, valid_o);
    end

    activation_i = 7;
    weight_i = 8;
    psum_i = 99;
    compute_en = 1'b1;
    clear_acc = 1'b1;
    #1;
    if (psum_o !== 0 || valid_o !== 1'b0) begin
      $fatal(1, "PE test 3 failed: expected (0, 0), got (%0d, %0d)", psum_o, valid_o);
    end

    $display("processing_element_tb passed");
    $finish;
  end
endmodule
"""


def _systolic_tile_template(operand_width: int, acc_width: int, seed_rows: int, seed_cols: int) -> str:
    return f"""module systolic_tile #(
  parameter int DATA_WIDTH = {operand_width},
  parameter int ACC_WIDTH = {acc_width},
  parameter int ROWS = {seed_rows},
  parameter int COLS = {seed_cols}
) (
  input  logic clk,
  input  logic rst_n,
  input  logic signed [ROWS*DATA_WIDTH-1:0] activations_west_i,
  input  logic signed [COLS*DATA_WIDTH-1:0] weights_north_i,
  input  logic load_inputs_en,
  input  logic compute_en,
  input  logic flush_pipeline,
  input  logic clear_acc,
  output logic signed [ROWS*COLS*ACC_WIDTH-1:0] psums_o,
  output logic [ROWS*COLS-1:0] valids_o
);
  logic signed [DATA_WIDTH-1:0] activation_regs [0:ROWS-1][0:COLS-1];
  logic signed [DATA_WIDTH-1:0] weight_regs [0:ROWS-1][0:COLS-1];
  logic signed [ACC_WIDTH-1:0] psum_regs [0:ROWS-1][0:COLS-1];
  logic signed [ACC_WIDTH-1:0] pe_psums [0:ROWS-1][0:COLS-1];
  logic pe_valids [0:ROWS-1][0:COLS-1];
  logic valid_regs [0:ROWS-1][0:COLS-1];

  genvar row_g;
  genvar col_g;
  generate
    for (row_g = 0; row_g < ROWS; row_g = row_g + 1) begin : gen_row
      for (col_g = 0; col_g < COLS; col_g = col_g + 1) begin : gen_col
        processing_element #(
          .DATA_WIDTH(DATA_WIDTH),
          .ACC_WIDTH(ACC_WIDTH)
        ) pe_inst (
          .activation_i(activation_regs[row_g][col_g]),
          .weight_i(weight_regs[row_g][col_g]),
          .psum_i(psum_regs[row_g][col_g]),
          .compute_en(compute_en),
          .clear_acc(clear_acc),
          .psum_o(pe_psums[row_g][col_g]),
          .valid_o(pe_valids[row_g][col_g])
        );
      end
    end
  endgenerate

  integer row_idx;
  integer col_idx;
  integer flat_idx;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      for (row_idx = 0; row_idx < ROWS; row_idx = row_idx + 1) begin
        for (col_idx = 0; col_idx < COLS; col_idx = col_idx + 1) begin
          activation_regs[row_idx][col_idx] <= '0;
          weight_regs[row_idx][col_idx] <= '0;
          psum_regs[row_idx][col_idx] <= '0;
          valid_regs[row_idx][col_idx] <= 1'b0;
        end
      end
    end else begin
      if (flush_pipeline) begin
        for (row_idx = 0; row_idx < ROWS; row_idx = row_idx + 1) begin
          for (col_idx = 0; col_idx < COLS; col_idx = col_idx + 1) begin
            activation_regs[row_idx][col_idx] <= '0;
            weight_regs[row_idx][col_idx] <= '0;
            valid_regs[row_idx][col_idx] <= 1'b0;
          end
        end
      end else if (load_inputs_en) begin
        for (row_idx = 0; row_idx < ROWS; row_idx = row_idx + 1) begin
          for (col_idx = COLS - 1; col_idx >= 0; col_idx = col_idx - 1) begin
            if (col_idx == 0) begin
              activation_regs[row_idx][col_idx] <= $signed(
                activations_west_i[row_idx*DATA_WIDTH +: DATA_WIDTH]
              );
            end else begin
              activation_regs[row_idx][col_idx] <= activation_regs[row_idx][col_idx - 1];
            end
          end
        end

        for (row_idx = ROWS - 1; row_idx >= 0; row_idx = row_idx - 1) begin
          for (col_idx = 0; col_idx < COLS; col_idx = col_idx + 1) begin
            if (row_idx == 0) begin
              weight_regs[row_idx][col_idx] <= $signed(
                weights_north_i[col_idx*DATA_WIDTH +: DATA_WIDTH]
              );
            end else begin
              weight_regs[row_idx][col_idx] <= weight_regs[row_idx - 1][col_idx];
            end
          end
        end
      end

      if (clear_acc) begin
        for (row_idx = 0; row_idx < ROWS; row_idx = row_idx + 1) begin
          for (col_idx = 0; col_idx < COLS; col_idx = col_idx + 1) begin
            psum_regs[row_idx][col_idx] <= '0;
            valid_regs[row_idx][col_idx] <= 1'b0;
          end
        end
      end else if (compute_en && !flush_pipeline) begin
        for (row_idx = 0; row_idx < ROWS; row_idx = row_idx + 1) begin
          for (col_idx = 0; col_idx < COLS; col_idx = col_idx + 1) begin
            psum_regs[row_idx][col_idx] <= pe_psums[row_idx][col_idx];
            valid_regs[row_idx][col_idx] <= pe_valids[row_idx][col_idx];
          end
        end
      end else if (!flush_pipeline) begin
        for (row_idx = 0; row_idx < ROWS; row_idx = row_idx + 1) begin
          for (col_idx = 0; col_idx < COLS; col_idx = col_idx + 1) begin
            valid_regs[row_idx][col_idx] <= 1'b0;
          end
        end
      end
    end
  end

  always_comb begin
    psums_o = '0;
    valids_o = '0;
    for (row_idx = 0; row_idx < ROWS; row_idx = row_idx + 1) begin
      for (col_idx = 0; col_idx < COLS; col_idx = col_idx + 1) begin
        flat_idx = (row_idx * COLS) + col_idx;
        psums_o[flat_idx*ACC_WIDTH +: ACC_WIDTH] = psum_regs[row_idx][col_idx];
        valids_o[flat_idx] = valid_regs[row_idx][col_idx];
      end
    end
  end
endmodule
"""


def _systolic_tile_tb_template(operand_width: int, acc_width: int) -> str:
    return f"""module systolic_tile_tb;
  localparam int DATA_WIDTH = {operand_width};
  localparam int ACC_WIDTH = {acc_width};
  localparam int ROWS = 2;
  localparam int COLS = 2;
  localparam int PE_COUNT = ROWS * COLS;

  logic clk;
  logic rst_n;
  logic signed [ROWS*DATA_WIDTH-1:0] activations_west_i;
  logic signed [COLS*DATA_WIDTH-1:0] weights_north_i;
  logic load_inputs_en;
  logic compute_en;
  logic flush_pipeline;
  logic clear_acc;
  logic signed [PE_COUNT*ACC_WIDTH-1:0] psums_o;
  logic [PE_COUNT-1:0] valids_o;

  systolic_tile #(
    .DATA_WIDTH(DATA_WIDTH),
    .ACC_WIDTH(ACC_WIDTH),
    .ROWS(ROWS),
    .COLS(COLS)
  ) dut (
    .clk(clk),
    .rst_n(rst_n),
    .activations_west_i(activations_west_i),
    .weights_north_i(weights_north_i),
    .load_inputs_en(load_inputs_en),
    .compute_en(compute_en),
    .flush_pipeline(flush_pipeline),
    .clear_acc(clear_acc),
    .psums_o(psums_o),
    .valids_o(valids_o)
  );

  always #5 clk = ~clk;

  task automatic set_edge_inputs(
    input integer row0_act,
    input integer row1_act,
    input integer col0_weight,
    input integer col1_weight
  );
    begin
      activations_west_i = '0;
      weights_north_i = '0;
      activations_west_i[0 +: DATA_WIDTH] = row0_act;
      activations_west_i[DATA_WIDTH +: DATA_WIDTH] = row1_act;
      weights_north_i[0 +: DATA_WIDTH] = col0_weight;
      weights_north_i[DATA_WIDTH +: DATA_WIDTH] = col1_weight;
    end
  endtask

  task automatic expect_outputs(
    input integer p0,
    input integer p1,
    input integer p2,
    input integer p3,
    input logic [PE_COUNT-1:0] expected_valids
  );
    begin
      if ($signed(psums_o[0 +: ACC_WIDTH]) !== p0 ||
          $signed(psums_o[ACC_WIDTH +: ACC_WIDTH]) !== p1 ||
          $signed(psums_o[2*ACC_WIDTH +: ACC_WIDTH]) !== p2 ||
          $signed(psums_o[3*ACC_WIDTH +: ACC_WIDTH]) !== p3 ||
          valids_o !== expected_valids) begin
        $fatal(
          1,
          "Tile expectation failed: expected (%0d %0d %0d %0d %0b), got (%0d %0d %0d %0d %0b)",
          p0, p1, p2, p3, expected_valids,
          $signed(psums_o[0 +: ACC_WIDTH]),
          $signed(psums_o[ACC_WIDTH +: ACC_WIDTH]),
          $signed(psums_o[2*ACC_WIDTH +: ACC_WIDTH]),
          $signed(psums_o[3*ACC_WIDTH +: ACC_WIDTH]),
          valids_o
        );
      end
    end
  endtask

  initial begin
    clk = 1'b0;
    rst_n = 1'b0;
    activations_west_i = '0;
    weights_north_i = '0;
    load_inputs_en = 1'b0;
    compute_en = 1'b0;
    flush_pipeline = 1'b0;
    clear_acc = 1'b0;
    repeat (2) @(posedge clk);
    rst_n = 1'b1;

    set_edge_inputs(1, 2, 5, 6);
    load_inputs_en = 1'b1;
    compute_en = 1'b0;
    clear_acc = 1'b0;
    @(posedge clk);
    #1;
    expect_outputs(0, 0, 0, 0, 4'b0000);

    set_edge_inputs(3, 4, 7, 8);
    load_inputs_en = 1'b1;
    compute_en = 1'b0;
    @(posedge clk);
    #1;
    expect_outputs(0, 0, 0, 0, 4'b0000);

    set_edge_inputs(0, 0, 0, 0);
    load_inputs_en = 1'b0;
    compute_en = 1'b1;
    @(posedge clk);
    #1;
    expect_outputs(21, 8, 20, 12, 4'b1111);

    load_inputs_en = 1'b0;
    compute_en = 1'b1;
    @(posedge clk);
    #1;
    expect_outputs(42, 16, 40, 24, 4'b1111);

    compute_en = 1'b0;
    flush_pipeline = 1'b1;
    @(posedge clk);
    #1;
    expect_outputs(42, 16, 40, 24, 4'b0000);

    flush_pipeline = 1'b0;
    compute_en = 1'b0;
    clear_acc = 1'b1;
    @(posedge clk);
    #1;
    expect_outputs(0, 0, 0, 0, 4'b0000);

    clear_acc = 1'b0;
    set_edge_inputs(9, 10, 1, 2);
    load_inputs_en = 1'b1;
    @(posedge clk);
    #1;
    expect_outputs(0, 0, 0, 0, 4'b0000);

    load_inputs_en = 1'b0;
    compute_en = 1'b1;
    @(posedge clk);
    #1;
    expect_outputs(9, 0, 0, 0, 4'b1111);

    $display("systolic_tile_tb passed");
    $finish;
  end
endmodule
"""


def _systolic_tile_rect_tb_template(operand_width: int, acc_width: int) -> str:
    rect_case = _systolic_tile_rectangular_flush_case()
    return f"""module systolic_tile_rect_tb;
  localparam int DATA_WIDTH = {operand_width};
  localparam int ACC_WIDTH = {acc_width};
  localparam int ROWS = 1;
  localparam int COLS = 3;
  localparam int PE_COUNT = ROWS * COLS;

  logic clk;
  logic rst_n;
  logic signed [ROWS*DATA_WIDTH-1:0] activations_west_i;
  logic signed [COLS*DATA_WIDTH-1:0] weights_north_i;
  logic load_inputs_en;
  logic compute_en;
  logic flush_pipeline;
  logic clear_acc;
  logic signed [PE_COUNT*ACC_WIDTH-1:0] psums_o;
  logic [PE_COUNT-1:0] valids_o;

  systolic_tile #(
    .DATA_WIDTH(DATA_WIDTH),
    .ACC_WIDTH(ACC_WIDTH),
    .ROWS(ROWS),
    .COLS(COLS)
  ) dut (
    .clk(clk),
    .rst_n(rst_n),
    .activations_west_i(activations_west_i),
    .weights_north_i(weights_north_i),
    .load_inputs_en(load_inputs_en),
    .compute_en(compute_en),
    .flush_pipeline(flush_pipeline),
    .clear_acc(clear_acc),
    .psums_o(psums_o),
    .valids_o(valids_o)
  );

  always #5 clk = ~clk;

  task automatic drive_and_expect(
    input integer row0_act,
    input integer col0_weight,
    input integer col1_weight,
    input integer col2_weight,
    input logic request_load,
    input logic request_compute,
    input logic request_flush,
    input logic request_clear,
    input integer p0,
    input integer p1,
    input integer p2,
    input logic [PE_COUNT-1:0] expected_valids
  );
    begin
      activations_west_i = '0;
      weights_north_i = '0;
      activations_west_i[0 +: DATA_WIDTH] = row0_act;
      weights_north_i[0 +: DATA_WIDTH] = col0_weight;
      weights_north_i[DATA_WIDTH +: DATA_WIDTH] = col1_weight;
      weights_north_i[2*DATA_WIDTH +: DATA_WIDTH] = col2_weight;
      load_inputs_en = request_load;
      compute_en = request_compute;
      flush_pipeline = request_flush;
      clear_acc = request_clear;
      @(posedge clk);
      #1;
      if ($signed(psums_o[0 +: ACC_WIDTH]) !== p0 ||
          $signed(psums_o[ACC_WIDTH +: ACC_WIDTH]) !== p1 ||
          $signed(psums_o[2*ACC_WIDTH +: ACC_WIDTH]) !== p2 ||
          valids_o !== expected_valids) begin
        $fatal(
          1,
          "systolic_tile_rect_tb failed: expected (%0d %0d %0d %0b), got (%0d %0d %0d %0b)",
          p0, p1, p2, expected_valids,
          $signed(psums_o[0 +: ACC_WIDTH]),
          $signed(psums_o[ACC_WIDTH +: ACC_WIDTH]),
          $signed(psums_o[2*ACC_WIDTH +: ACC_WIDTH]),
          valids_o
        );
      end
    end
  endtask

  initial begin
    clk = 1'b0;
    rst_n = 1'b0;
    activations_west_i = '0;
    weights_north_i = '0;
    load_inputs_en = 1'b0;
    compute_en = 1'b0;
    flush_pipeline = 1'b0;
    clear_acc = 1'b0;
    repeat (2) @(posedge clk);
    rst_n = 1'b1;

{_render_systolic_tile_rect_tb_steps(rect_case)}

    $display("systolic_tile_rect_tb passed");
    $finish;
  end
endmodule
"""


def _scratchpad_controller_template(operand_width: int, seed_rows: int, seed_cols: int) -> str:
    return f"""module scratchpad_controller #(
  parameter int DATA_WIDTH = {operand_width},
  parameter int ROWS = {seed_rows},
  parameter int COLS = {seed_cols},
  parameter int DEPTH = 4,
  parameter int BANK_COUNT = 2,
  parameter int ADDR_WIDTH = $clog2(DEPTH),
  parameter int BANK_SEL_WIDTH = (BANK_COUNT > 1) ? $clog2(BANK_COUNT) : 1
) (
  input  logic clk,
  input  logic rst_n,
  input  logic write_activations_en_i,
  input  logic [BANK_SEL_WIDTH-1:0] activation_write_bank_i,
  input  logic [ADDR_WIDTH-1:0] activation_write_addr_i,
  input  logic signed [ROWS*DATA_WIDTH-1:0] activations_write_data_i,
  input  logic write_weights_en_i,
  input  logic [BANK_SEL_WIDTH-1:0] weight_write_bank_i,
  input  logic [ADDR_WIDTH-1:0] weight_write_addr_i,
  input  logic signed [COLS*DATA_WIDTH-1:0] weights_write_data_i,
  input  logic load_vector_en_i,
  input  logic [BANK_SEL_WIDTH-1:0] activation_read_bank_i,
  input  logic [ADDR_WIDTH-1:0] activation_read_addr_i,
  input  logic [BANK_SEL_WIDTH-1:0] weight_read_bank_i,
  input  logic [ADDR_WIDTH-1:0] weight_read_addr_i,
  output logic signed [ROWS*DATA_WIDTH-1:0] activations_west_o,
  output logic signed [COLS*DATA_WIDTH-1:0] weights_north_o,
  output logic vector_valid_o
);
  logic signed [ROWS*DATA_WIDTH-1:0] activation_bank [0:BANK_COUNT-1][0:DEPTH-1];
  logic signed [COLS*DATA_WIDTH-1:0] weight_bank [0:BANK_COUNT-1][0:DEPTH-1];
  logic activation_valid_bank [0:BANK_COUNT-1][0:DEPTH-1];
  logic weight_valid_bank [0:BANK_COUNT-1][0:DEPTH-1];
  integer bank_sel_idx;
  integer bank_idx;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      for (bank_sel_idx = 0; bank_sel_idx < BANK_COUNT; bank_sel_idx = bank_sel_idx + 1) begin
        for (bank_idx = 0; bank_idx < DEPTH; bank_idx = bank_idx + 1) begin
          activation_bank[bank_sel_idx][bank_idx] <= '0;
          weight_bank[bank_sel_idx][bank_idx] <= '0;
          activation_valid_bank[bank_sel_idx][bank_idx] <= 1'b0;
          weight_valid_bank[bank_sel_idx][bank_idx] <= 1'b0;
        end
      end
    end else begin
      if (write_activations_en_i) begin
        activation_bank[activation_write_bank_i][activation_write_addr_i] <= activations_write_data_i;
        activation_valid_bank[activation_write_bank_i][activation_write_addr_i] <= 1'b1;
      end
      if (write_weights_en_i) begin
        weight_bank[weight_write_bank_i][weight_write_addr_i] <= weights_write_data_i;
        weight_valid_bank[weight_write_bank_i][weight_write_addr_i] <= 1'b1;
      end
    end
  end

  always_comb begin
    activations_west_o = activation_bank[activation_read_bank_i][activation_read_addr_i];
    weights_north_o = weight_bank[weight_read_bank_i][weight_read_addr_i];
    vector_valid_o = load_vector_en_i &&
      activation_valid_bank[activation_read_bank_i][activation_read_addr_i] &&
      weight_valid_bank[weight_read_bank_i][weight_read_addr_i];
  end
endmodule
"""


def _dma_engine_template(operand_width: int, seed_rows: int, seed_cols: int) -> str:
    return f"""module dma_engine #(
  parameter int DATA_WIDTH = {operand_width},
  parameter int ROWS = {seed_rows},
  parameter int COLS = {seed_cols},
  parameter int DEPTH = 4,
  parameter int ADDR_WIDTH = $clog2(DEPTH),
  parameter int MAX_DIM = (ROWS > COLS) ? ROWS : COLS
) (
  input  logic clk,
  input  logic rst_n,
  input  logic dma_valid_i,
  input  logic dma_write_weights_i,
  input  logic [ADDR_WIDTH-1:0] dma_addr_i,
  input  logic signed [MAX_DIM*DATA_WIDTH-1:0] dma_payload_i,
  output logic write_activations_en_o,
  output logic [ADDR_WIDTH-1:0] activation_write_addr_o,
  output logic signed [ROWS*DATA_WIDTH-1:0] activations_write_data_o,
  output logic write_weights_en_o,
  output logic [ADDR_WIDTH-1:0] weight_write_addr_o,
  output logic signed [COLS*DATA_WIDTH-1:0] weights_write_data_o,
  output logic dma_done_o,
  output logic dma_busy_o
);
  logic pending_valid_q;
  logic pending_write_weights_q;
  logic [ADDR_WIDTH-1:0] pending_addr_q;
  logic signed [MAX_DIM*DATA_WIDTH-1:0] pending_payload_q;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      pending_valid_q <= 1'b0;
      pending_write_weights_q <= 1'b0;
      pending_addr_q <= '0;
      pending_payload_q <= '0;
    end else begin
      pending_valid_q <= dma_valid_i;
      if (dma_valid_i) begin
        pending_write_weights_q <= dma_write_weights_i;
        pending_addr_q <= dma_addr_i;
        pending_payload_q <= dma_payload_i;
      end
    end
  end

  always_comb begin
    write_activations_en_o = pending_valid_q && !pending_write_weights_q;
    activation_write_addr_o = pending_addr_q;
    activations_write_data_o = pending_payload_q[0 +: ROWS*DATA_WIDTH];

    write_weights_en_o = pending_valid_q && pending_write_weights_q;
    weight_write_addr_o = pending_addr_q;
    weights_write_data_o = pending_payload_q[0 +: COLS*DATA_WIDTH];

    dma_done_o = pending_valid_q;
    dma_busy_o = pending_valid_q || dma_valid_i;
  end
endmodule
"""


def _dma_engine_tb_template(operand_width: int) -> str:
    return f"""module dma_engine_tb;
  localparam int DATA_WIDTH = {operand_width};
  localparam int ROWS = 2;
  localparam int COLS = 2;
  localparam int DEPTH = 4;
  localparam int ADDR_WIDTH = $clog2(DEPTH);
  localparam int MAX_DIM = (ROWS > COLS) ? ROWS : COLS;

  logic clk;
  logic rst_n;
  logic dma_valid_i;
  logic dma_write_weights_i;
  logic [ADDR_WIDTH-1:0] dma_addr_i;
  logic signed [MAX_DIM*DATA_WIDTH-1:0] dma_payload_i;
  logic write_activations_en_o;
  logic [ADDR_WIDTH-1:0] activation_write_addr_o;
  logic signed [ROWS*DATA_WIDTH-1:0] activations_write_data_o;
  logic write_weights_en_o;
  logic [ADDR_WIDTH-1:0] weight_write_addr_o;
  logic signed [COLS*DATA_WIDTH-1:0] weights_write_data_o;
  logic dma_done_o;
  logic dma_busy_o;

  dma_engine #(
    .DATA_WIDTH(DATA_WIDTH),
    .ROWS(ROWS),
    .COLS(COLS),
    .DEPTH(DEPTH)
  ) dut (
    .clk(clk),
    .rst_n(rst_n),
    .dma_valid_i(dma_valid_i),
    .dma_write_weights_i(dma_write_weights_i),
    .dma_addr_i(dma_addr_i),
    .dma_payload_i(dma_payload_i),
    .write_activations_en_o(write_activations_en_o),
    .activation_write_addr_o(activation_write_addr_o),
    .activations_write_data_o(activations_write_data_o),
    .write_weights_en_o(write_weights_en_o),
    .weight_write_addr_o(weight_write_addr_o),
    .weights_write_data_o(weights_write_data_o),
    .dma_done_o(dma_done_o),
    .dma_busy_o(dma_busy_o)
  );

  always #5 clk = ~clk;

  task automatic idle_dma;
    begin
      dma_valid_i = 1'b0;
      dma_write_weights_i = 1'b0;
      dma_addr_i = '0;
      dma_payload_i = '0;
    end
  endtask

  task automatic issue_dma(
    input logic write_weights,
    input integer addr,
    input integer lane0,
    input integer lane1
  );
    begin
      idle_dma();
      dma_valid_i = 1'b1;
      dma_write_weights_i = write_weights;
      dma_addr_i = addr[ADDR_WIDTH-1:0];
      dma_payload_i[0 +: DATA_WIDTH] = lane0;
      dma_payload_i[DATA_WIDTH +: DATA_WIDTH] = lane1;
      @(posedge clk);
      #1;
    end
  endtask

  task automatic expect_idle_outputs;
    begin
      if (write_activations_en_o !== 1'b0 ||
          write_weights_en_o !== 1'b0 ||
          dma_done_o !== 1'b0) begin
        $fatal(1, "dma_engine_tb idle expectation failed");
      end
    end
  endtask

  task automatic expect_activation_write(
    input integer addr,
    input integer lane0,
    input integer lane1
  );
    begin
      if (write_activations_en_o !== 1'b1 ||
          activation_write_addr_o !== addr[ADDR_WIDTH-1:0] ||
          $signed(activations_write_data_o[0 +: DATA_WIDTH]) !== lane0 ||
          $signed(activations_write_data_o[DATA_WIDTH +: DATA_WIDTH]) !== lane1 ||
          write_weights_en_o !== 1'b0 ||
          dma_done_o !== 1'b1) begin
        $fatal(1, "dma_engine_tb activation write expectation failed");
      end
    end
  endtask

  task automatic expect_weight_write(
    input integer addr,
    input integer lane0,
    input integer lane1
  );
    begin
      if (write_weights_en_o !== 1'b1 ||
          weight_write_addr_o !== addr[ADDR_WIDTH-1:0] ||
          $signed(weights_write_data_o[0 +: DATA_WIDTH]) !== lane0 ||
          $signed(weights_write_data_o[DATA_WIDTH +: DATA_WIDTH]) !== lane1 ||
          write_activations_en_o !== 1'b0 ||
          dma_done_o !== 1'b1) begin
        $fatal(1, "dma_engine_tb weight write expectation failed");
      end
    end
  endtask

  initial begin
    clk = 1'b0;
    rst_n = 1'b0;
    idle_dma();
    repeat (2) @(posedge clk);
    rst_n = 1'b1;

    issue_dma(1'b0, 1, 3, 4);
    expect_activation_write(1, 3, 4);
    idle_dma();
    @(posedge clk);
    #1;
    expect_idle_outputs();

    issue_dma(1'b1, 2, 7, 8);
    expect_weight_write(2, 7, 8);
    idle_dma();
    @(posedge clk);
    #1;
    expect_idle_outputs();

    $display("dma_engine_tb passed");
    $finish;
  end
endmodule
"""


def _scratchpad_controller_tb_template(operand_width: int) -> str:
    return f"""module scratchpad_controller_tb;
  localparam int DATA_WIDTH = {operand_width};
  localparam int ROWS = 2;
  localparam int COLS = 2;
  localparam int DEPTH = 4;
  localparam int BANK_COUNT = 2;
  localparam int ADDR_WIDTH = $clog2(DEPTH);
  localparam int BANK_SEL_WIDTH = (BANK_COUNT > 1) ? $clog2(BANK_COUNT) : 1;

  logic clk;
  logic rst_n;
  logic write_activations_en_i;
  logic [BANK_SEL_WIDTH-1:0] activation_write_bank_i;
  logic [ADDR_WIDTH-1:0] activation_write_addr_i;
  logic signed [ROWS*DATA_WIDTH-1:0] activations_write_data_i;
  logic write_weights_en_i;
  logic [BANK_SEL_WIDTH-1:0] weight_write_bank_i;
  logic [ADDR_WIDTH-1:0] weight_write_addr_i;
  logic signed [COLS*DATA_WIDTH-1:0] weights_write_data_i;
  logic load_vector_en_i;
  logic [BANK_SEL_WIDTH-1:0] activation_read_bank_i;
  logic [ADDR_WIDTH-1:0] activation_read_addr_i;
  logic [BANK_SEL_WIDTH-1:0] weight_read_bank_i;
  logic [ADDR_WIDTH-1:0] weight_read_addr_i;
  logic signed [ROWS*DATA_WIDTH-1:0] activations_west_o;
  logic signed [COLS*DATA_WIDTH-1:0] weights_north_o;
  logic vector_valid_o;

  scratchpad_controller #(
    .DATA_WIDTH(DATA_WIDTH),
    .ROWS(ROWS),
    .COLS(COLS),
    .DEPTH(DEPTH),
    .BANK_COUNT(BANK_COUNT)
  ) dut (
    .clk(clk),
    .rst_n(rst_n),
    .write_activations_en_i(write_activations_en_i),
    .activation_write_bank_i(activation_write_bank_i),
    .activation_write_addr_i(activation_write_addr_i),
    .activations_write_data_i(activations_write_data_i),
    .write_weights_en_i(write_weights_en_i),
    .weight_write_bank_i(weight_write_bank_i),
    .weight_write_addr_i(weight_write_addr_i),
    .weights_write_data_i(weights_write_data_i),
    .load_vector_en_i(load_vector_en_i),
    .activation_read_bank_i(activation_read_bank_i),
    .activation_read_addr_i(activation_read_addr_i),
    .weight_read_bank_i(weight_read_bank_i),
    .weight_read_addr_i(weight_read_addr_i),
    .activations_west_o(activations_west_o),
    .weights_north_o(weights_north_o),
    .vector_valid_o(vector_valid_o)
  );

  always #5 clk = ~clk;

  task automatic idle_controls;
    begin
      write_activations_en_i = 1'b0;
      write_weights_en_i = 1'b0;
      load_vector_en_i = 1'b0;
      activation_write_bank_i = '0;
      weight_write_bank_i = '0;
      activation_read_bank_i = '0;
      weight_read_bank_i = '0;
      activations_write_data_i = '0;
      weights_write_data_i = '0;
    end
  endtask

  task automatic write_activation_vector(
    input integer bank,
    input integer addr,
    input integer row0,
    input integer row1
  );
    begin
      idle_controls();
      activation_write_bank_i = bank[BANK_SEL_WIDTH-1:0];
      activation_write_addr_i = addr[ADDR_WIDTH-1:0];
      activations_write_data_i[0 +: DATA_WIDTH] = row0;
      activations_write_data_i[DATA_WIDTH +: DATA_WIDTH] = row1;
      write_activations_en_i = 1'b1;
      @(posedge clk);
      #1;
    end
  endtask

  task automatic write_weight_vector(
    input integer bank,
    input integer addr,
    input integer col0,
    input integer col1
  );
    begin
      idle_controls();
      weight_write_bank_i = bank[BANK_SEL_WIDTH-1:0];
      weight_write_addr_i = addr[ADDR_WIDTH-1:0];
      weights_write_data_i[0 +: DATA_WIDTH] = col0;
      weights_write_data_i[DATA_WIDTH +: DATA_WIDTH] = col1;
      write_weights_en_i = 1'b1;
      @(posedge clk);
      #1;
    end
  endtask

  task automatic load_and_expect(
    input integer act_bank,
    input integer act_addr,
    input integer weight_bank,
    input integer weight_addr,
    input integer act0,
    input integer act1,
    input integer w0,
    input integer w1,
    input logic request_load,
    input logic expected_valid
  );
    begin
      idle_controls();
      activation_read_bank_i = act_bank[BANK_SEL_WIDTH-1:0];
      activation_read_addr_i = act_addr[ADDR_WIDTH-1:0];
      weight_read_bank_i = weight_bank[BANK_SEL_WIDTH-1:0];
      weight_read_addr_i = weight_addr[ADDR_WIDTH-1:0];
      load_vector_en_i = request_load;
      @(posedge clk);
      #1;
      if ($signed(activations_west_o[0 +: DATA_WIDTH]) !== act0 ||
          $signed(activations_west_o[DATA_WIDTH +: DATA_WIDTH]) !== act1 ||
          $signed(weights_north_o[0 +: DATA_WIDTH]) !== w0 ||
          $signed(weights_north_o[DATA_WIDTH +: DATA_WIDTH]) !== w1 ||
          vector_valid_o !== expected_valid) begin
        $fatal(1, "scratchpad_controller_tb failed");
      end
    end
  endtask

  initial begin
    clk = 1'b0;
    rst_n = 1'b0;
    activation_read_addr_i = '0;
    weight_read_addr_i = '0;
    idle_controls();
    repeat (2) @(posedge clk);
    rst_n = 1'b1;

    write_activation_vector(0, 0, 1, 2);
    write_weight_vector(0, 0, 5, 6);
    write_activation_vector(1, 0, 9, 10);
    write_weight_vector(1, 0, 11, 12);

    load_and_expect(0, 0, 0, 0, 1, 2, 5, 6, 1'b1, 1'b1);
    load_and_expect(1, 0, 1, 0, 9, 10, 11, 12, 1'b1, 1'b1);
    load_and_expect(1, 0, 1, 0, 9, 10, 11, 12, 1'b0, 1'b0);
    load_and_expect(1, 1, 1, 1, 0, 0, 0, 0, 1'b1, 1'b0);

    $display("scratchpad_controller_tb passed");
    $finish;
  end
endmodule
"""


def _tile_compute_unit_template(operand_width: int, acc_width: int, seed_rows: int, seed_cols: int) -> str:
    return f"""module tile_compute_unit #(
  parameter int DATA_WIDTH = {operand_width},
  parameter int ACC_WIDTH = {acc_width},
  parameter int ROWS = {seed_rows},
  parameter int COLS = {seed_cols},
  parameter int DEPTH = 4,
  parameter int BANK_COUNT = 2,
  parameter int ADDR_WIDTH = $clog2(DEPTH),
  parameter int BANK_SEL_WIDTH = (BANK_COUNT > 1) ? $clog2(BANK_COUNT) : 1,
  parameter int MAX_DIM = (ROWS > COLS) ? ROWS : COLS
) (
  input  logic clk,
  input  logic rst_n,
  input  logic dma_valid_i,
  input  logic dma_write_weights_i,
  input  logic [ADDR_WIDTH-1:0] dma_addr_i,
  input  logic signed [MAX_DIM*DATA_WIDTH-1:0] dma_payload_i,
  input  logic [BANK_SEL_WIDTH-1:0] activation_write_bank_i,
  input  logic [BANK_SEL_WIDTH-1:0] weight_write_bank_i,
  input  logic load_vector_en_i,
  input  logic [BANK_SEL_WIDTH-1:0] activation_read_bank_i,
  input  logic [ADDR_WIDTH-1:0] activation_read_addr_i,
  input  logic [BANK_SEL_WIDTH-1:0] weight_read_bank_i,
  input  logic [ADDR_WIDTH-1:0] weight_read_addr_i,
  input  logic compute_en_i,
  input  logic flush_pipeline_i,
  input  logic clear_acc_i,
  output logic scratchpad_vector_valid_o,
  output logic dma_done_o,
  output logic dma_busy_o,
  output logic signed [ROWS*COLS*ACC_WIDTH-1:0] psums_o,
  output logic [ROWS*COLS-1:0] valids_o
);
  logic signed [ROWS*DATA_WIDTH-1:0] activations_west;
  logic signed [COLS*DATA_WIDTH-1:0] weights_north;
  logic write_activations_en;
  logic [ADDR_WIDTH-1:0] activation_write_addr;
  logic signed [ROWS*DATA_WIDTH-1:0] activations_write_data;
  logic write_weights_en;
  logic [ADDR_WIDTH-1:0] weight_write_addr;
  logic signed [COLS*DATA_WIDTH-1:0] weights_write_data;
  logic [BANK_SEL_WIDTH-1:0] activation_write_bank_q;
  logic [BANK_SEL_WIDTH-1:0] weight_write_bank_q;

  dma_engine #(
    .DATA_WIDTH(DATA_WIDTH),
    .ROWS(ROWS),
    .COLS(COLS),
    .DEPTH(DEPTH)
  ) dma_inst (
    .clk(clk),
    .rst_n(rst_n),
    .dma_valid_i(dma_valid_i),
    .dma_write_weights_i(dma_write_weights_i),
    .dma_addr_i(dma_addr_i),
    .dma_payload_i(dma_payload_i),
    .write_activations_en_o(write_activations_en),
    .activation_write_addr_o(activation_write_addr),
    .activations_write_data_o(activations_write_data),
    .write_weights_en_o(write_weights_en),
    .weight_write_addr_o(weight_write_addr),
    .weights_write_data_o(weights_write_data),
    .dma_done_o(dma_done_o),
    .dma_busy_o(dma_busy_o)
  );

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      activation_write_bank_q <= '0;
      weight_write_bank_q <= '0;
    end else if (dma_valid_i) begin
      activation_write_bank_q <= activation_write_bank_i;
      weight_write_bank_q <= weight_write_bank_i;
    end
  end

  scratchpad_controller #(
    .DATA_WIDTH(DATA_WIDTH),
    .ROWS(ROWS),
    .COLS(COLS),
    .DEPTH(DEPTH),
    .BANK_COUNT(BANK_COUNT)
  ) scratchpad_inst (
    .clk(clk),
    .rst_n(rst_n),
    .write_activations_en_i(write_activations_en),
    .activation_write_bank_i(activation_write_bank_q),
    .activation_write_addr_i(activation_write_addr),
    .activations_write_data_i(activations_write_data),
    .write_weights_en_i(write_weights_en),
    .weight_write_bank_i(weight_write_bank_q),
    .weight_write_addr_i(weight_write_addr),
    .weights_write_data_i(weights_write_data),
    .load_vector_en_i(load_vector_en_i),
    .activation_read_bank_i(activation_read_bank_i),
    .activation_read_addr_i(activation_read_addr_i),
    .weight_read_bank_i(weight_read_bank_i),
    .weight_read_addr_i(weight_read_addr_i),
    .activations_west_o(activations_west),
    .weights_north_o(weights_north),
    .vector_valid_o(scratchpad_vector_valid_o)
  );

  systolic_tile #(
    .DATA_WIDTH(DATA_WIDTH),
    .ACC_WIDTH(ACC_WIDTH),
    .ROWS(ROWS),
    .COLS(COLS)
  ) tile_inst (
    .clk(clk),
    .rst_n(rst_n),
    .activations_west_i(activations_west),
    .weights_north_i(weights_north),
    .load_inputs_en(scratchpad_vector_valid_o),
    .compute_en(compute_en_i),
    .flush_pipeline(flush_pipeline_i),
    .clear_acc(clear_acc_i),
    .psums_o(psums_o),
    .valids_o(valids_o)
  );
endmodule
"""


def _tile_compute_unit_tb_template(operand_width: int, acc_width: int) -> str:
    return f"""module tile_compute_unit_tb;
  localparam int DATA_WIDTH = {operand_width};
  localparam int ACC_WIDTH = {acc_width};
  localparam int ROWS = 2;
  localparam int COLS = 2;
  localparam int DEPTH = 4;
  localparam int BANK_COUNT = 2;
  localparam int ADDR_WIDTH = $clog2(DEPTH);
  localparam int BANK_SEL_WIDTH = (BANK_COUNT > 1) ? $clog2(BANK_COUNT) : 1;
  localparam int MAX_DIM = (ROWS > COLS) ? ROWS : COLS;
  localparam int PE_COUNT = ROWS * COLS;

  logic clk;
  logic rst_n;
  logic dma_valid_i;
  logic dma_write_weights_i;
  logic [ADDR_WIDTH-1:0] dma_addr_i;
  logic signed [MAX_DIM*DATA_WIDTH-1:0] dma_payload_i;
  logic [BANK_SEL_WIDTH-1:0] activation_write_bank_i;
  logic [BANK_SEL_WIDTH-1:0] weight_write_bank_i;
  logic load_vector_en_i;
  logic [BANK_SEL_WIDTH-1:0] activation_read_bank_i;
  logic [ADDR_WIDTH-1:0] activation_read_addr_i;
  logic [BANK_SEL_WIDTH-1:0] weight_read_bank_i;
  logic [ADDR_WIDTH-1:0] weight_read_addr_i;
  logic compute_en_i;
  logic flush_pipeline_i;
  logic clear_acc_i;
  logic scratchpad_vector_valid_o;
  logic dma_done_o;
  logic dma_busy_o;
  logic signed [PE_COUNT*ACC_WIDTH-1:0] psums_o;
  logic [PE_COUNT-1:0] valids_o;

  tile_compute_unit #(
    .DATA_WIDTH(DATA_WIDTH),
    .ACC_WIDTH(ACC_WIDTH),
    .ROWS(ROWS),
    .COLS(COLS),
    .DEPTH(DEPTH),
    .BANK_COUNT(BANK_COUNT)
  ) dut (
    .clk(clk),
    .rst_n(rst_n),
    .dma_valid_i(dma_valid_i),
    .dma_write_weights_i(dma_write_weights_i),
    .dma_addr_i(dma_addr_i),
    .dma_payload_i(dma_payload_i),
    .activation_write_bank_i(activation_write_bank_i),
    .weight_write_bank_i(weight_write_bank_i),
    .load_vector_en_i(load_vector_en_i),
    .activation_read_bank_i(activation_read_bank_i),
    .activation_read_addr_i(activation_read_addr_i),
    .weight_read_bank_i(weight_read_bank_i),
    .weight_read_addr_i(weight_read_addr_i),
    .compute_en_i(compute_en_i),
    .flush_pipeline_i(flush_pipeline_i),
    .clear_acc_i(clear_acc_i),
    .scratchpad_vector_valid_o(scratchpad_vector_valid_o),
    .dma_done_o(dma_done_o),
    .dma_busy_o(dma_busy_o),
    .psums_o(psums_o),
    .valids_o(valids_o)
  );

  always #5 clk = ~clk;

  task automatic idle_controls;
    begin
      dma_valid_i = 1'b0;
      dma_write_weights_i = 1'b0;
      dma_addr_i = '0;
      dma_payload_i = '0;
      activation_write_bank_i = '0;
      weight_write_bank_i = '0;
      load_vector_en_i = 1'b0;
      activation_read_bank_i = '0;
      weight_read_bank_i = '0;
      compute_en_i = 1'b0;
      flush_pipeline_i = 1'b0;
      clear_acc_i = 1'b0;
    end
  endtask

  task automatic dma_activation_vector(
    input integer bank,
    input integer addr,
    input integer row0,
    input integer row1
  );
    begin
      idle_controls();
      dma_valid_i = 1'b1;
      dma_write_weights_i = 1'b0;
      activation_write_bank_i = bank[BANK_SEL_WIDTH-1:0];
      dma_addr_i = addr[ADDR_WIDTH-1:0];
      dma_payload_i[0 +: DATA_WIDTH] = row0;
      dma_payload_i[DATA_WIDTH +: DATA_WIDTH] = row1;
      @(posedge clk);
      #1;
    end
  endtask

  task automatic dma_weight_vector(
    input integer bank,
    input integer addr,
    input integer col0,
    input integer col1
  );
    begin
      idle_controls();
      dma_valid_i = 1'b1;
      dma_write_weights_i = 1'b1;
      weight_write_bank_i = bank[BANK_SEL_WIDTH-1:0];
      dma_addr_i = addr[ADDR_WIDTH-1:0];
      dma_payload_i[0 +: DATA_WIDTH] = col0;
      dma_payload_i[DATA_WIDTH +: DATA_WIDTH] = col1;
      @(posedge clk);
      #1;
    end
  endtask

  task automatic load_tile_vectors(
    input integer act_bank,
    input integer act_addr,
    input integer weight_bank,
    input integer weight_addr
  );
    begin
      idle_controls();
      activation_read_bank_i = act_bank[BANK_SEL_WIDTH-1:0];
      activation_read_addr_i = act_addr[ADDR_WIDTH-1:0];
      weight_read_bank_i = weight_bank[BANK_SEL_WIDTH-1:0];
      weight_read_addr_i = weight_addr[ADDR_WIDTH-1:0];
      load_vector_en_i = 1'b1;
      @(posedge clk);
      #1;
    end
  endtask

  task automatic compute_step;
    begin
      idle_controls();
      compute_en_i = 1'b1;
      @(posedge clk);
      #1;
    end
  endtask

  task automatic clear_step;
    begin
      idle_controls();
      clear_acc_i = 1'b1;
      @(posedge clk);
      #1;
    end
  endtask

  task automatic flush_step;
    begin
      idle_controls();
      flush_pipeline_i = 1'b1;
      @(posedge clk);
      #1;
    end
  endtask

  task automatic expect_outputs(
    input integer p0,
    input integer p1,
    input integer p2,
    input integer p3,
    input logic [PE_COUNT-1:0] expected_valids,
    input logic expected_scratchpad_valid
  );
    begin
      if ($signed(psums_o[0 +: ACC_WIDTH]) !== p0 ||
          $signed(psums_o[ACC_WIDTH +: ACC_WIDTH]) !== p1 ||
          $signed(psums_o[2*ACC_WIDTH +: ACC_WIDTH]) !== p2 ||
          $signed(psums_o[3*ACC_WIDTH +: ACC_WIDTH]) !== p3 ||
          valids_o !== expected_valids ||
          scratchpad_vector_valid_o !== expected_scratchpad_valid) begin
        $fatal(1, "tile_compute_unit_tb failed");
      end
    end
  endtask

  initial begin
    clk = 1'b0;
    rst_n = 1'b0;
    activation_read_addr_i = '0;
    weight_read_addr_i = '0;
    idle_controls();
    repeat (2) @(posedge clk);
    rst_n = 1'b1;

    dma_activation_vector(0, 0, 1, 2);
    idle_controls();
    @(posedge clk);
    #1;

    dma_weight_vector(0, 0, 5, 6);
    idle_controls();
    @(posedge clk);
    #1;

    load_tile_vectors(0, 0, 0, 0);
    expect_outputs(0, 0, 0, 0, 4'b0000, 1'b1);

    dma_activation_vector(1, 0, 3, 4);
    idle_controls();
    @(posedge clk);
    #1;

    dma_weight_vector(1, 0, 7, 8);
    idle_controls();
    @(posedge clk);
    #1;

    load_tile_vectors(1, 0, 1, 0);
    expect_outputs(0, 0, 0, 0, 4'b0000, 1'b1);

    compute_step();
    expect_outputs(21, 8, 20, 12, 4'b1111, 1'b0);

    compute_step();
    expect_outputs(42, 16, 40, 24, 4'b1111, 1'b0);

    flush_step();
    expect_outputs(42, 16, 40, 24, 4'b0000, 1'b0);

    clear_step();
    expect_outputs(0, 0, 0, 0, 4'b0000, 1'b0);

    $display("tile_compute_unit_tb passed");
    $finish;
  end
endmodule
"""


def _scheduler_template(operand_width: int, seed_rows: int, seed_cols: int) -> str:
    return f"""module scheduler #(
  parameter int DATA_WIDTH = {operand_width},
  parameter int ROWS = {seed_rows},
  parameter int COLS = {seed_cols},
  parameter int DEPTH = 4,
  parameter int ADDR_WIDTH = $clog2(DEPTH),
  parameter int MAX_DIM = (ROWS > COLS) ? ROWS : COLS
) (
  input  logic clk,
  input  logic rst_n,
  input  logic start_i,
  input  logic [1:0] slot_count_i,
  input  logic [1:0] load_iterations_i,
  input  logic [3:0] compute_iterations_i,
  input  logic [ADDR_WIDTH-1:0] activation_base_addr_i,
  input  logic [ADDR_WIDTH-1:0] weight_base_addr_i,
  input  logic [ADDR_WIDTH-1:0] result_base_addr_i,
  input  logic [ADDR_WIDTH-1:0] slot_stride_i,
  input  logic [ADDR_WIDTH-1:0] store_stride_i,
  input  logic [1:0] store_burst_count_i,
  input  logic clear_on_done_i,
  input  logic signed [ROWS*DATA_WIDTH-1:0] activation_slot0_i,
  input  logic signed [ROWS*DATA_WIDTH-1:0] activation_slot1_i,
  input  logic signed [COLS*DATA_WIDTH-1:0] weight_slot0_i,
  input  logic signed [COLS*DATA_WIDTH-1:0] weight_slot1_i,
  output logic dma_valid_o,
  output logic dma_write_weights_o,
  output logic [ADDR_WIDTH-1:0] dma_addr_o,
  output logic signed [MAX_DIM*DATA_WIDTH-1:0] dma_payload_o,
  output logic load_vector_en_o,
  output logic [ADDR_WIDTH-1:0] activation_read_addr_o,
  output logic [ADDR_WIDTH-1:0] weight_read_addr_o,
  output logic store_results_en_o,
  output logic [ADDR_WIDTH-1:0] result_write_addr_o,
  output logic [ADDR_WIDTH-1:0] store_burst_index_o,
  output logic compute_en_o,
  output logic flush_pipeline_o,
  output logic clear_acc_o,
  output logic busy_o,
  output logic done_o,
  output logic [3:0] state_o
);
  localparam logic [3:0] S_IDLE = 4'd0;
  localparam logic [3:0] S_DMA_ACT = 4'd1;
  localparam logic [3:0] S_DMA_WGT = 4'd2;
  localparam logic [3:0] S_LOAD = 4'd3;
  localparam logic [3:0] S_COMPUTE = 4'd4;
  localparam logic [3:0] S_STORE = 4'd5;
  localparam logic [3:0] S_FLUSH = 4'd6;
  localparam logic [3:0] S_CLEAR = 4'd7;
  localparam logic [3:0] S_DONE = 4'd8;
  localparam logic [ADDR_WIDTH-1:0] ADDR_ZERO = '0;

  logic [3:0] state_q;
  logic [3:0] state_d;
  logic [ADDR_WIDTH-1:0] slot_index_q;
  logic [ADDR_WIDTH-1:0] slot_index_d;
  logic [ADDR_WIDTH-1:0] load_index_q;
  logic [ADDR_WIDTH-1:0] load_index_d;
  logic [ADDR_WIDTH-1:0] store_index_q;
  logic [ADDR_WIDTH-1:0] store_index_d;
  logic [3:0] compute_count_q;
  logic [3:0] compute_count_d;
  logic [1:0] program_slot_count_q;
  logic [1:0] program_slot_count_d;
  logic [1:0] program_load_iterations_q;
  logic [1:0] program_load_iterations_d;
  logic [3:0] program_compute_iterations_q;
  logic [3:0] program_compute_iterations_d;
  logic program_clear_on_done_q;
  logic program_clear_on_done_d;
  logic [1:0] slot_count_sanitized;
  logic [1:0] load_iterations_sanitized;
  logic [1:0] store_burst_count_sanitized;
  logic [ADDR_WIDTH-1:0] slot_stride_sanitized;
  logic [ADDR_WIDTH-1:0] store_stride_sanitized;

  function automatic logic [ADDR_WIDTH-1:0] descriptor_addr(
    input logic [ADDR_WIDTH-1:0] base_addr,
    input logic [ADDR_WIDTH-1:0] index_value,
    input logic [ADDR_WIDTH-1:0] stride_value
  );
    begin
      descriptor_addr = base_addr + (index_value * stride_value);
    end
  endfunction

  always_comb begin
    if (slot_count_i == 2'd0) begin
      slot_count_sanitized = 2'd1;
    end else if (slot_count_i > 2'd2) begin
      slot_count_sanitized = 2'd2;
    end else begin
      slot_count_sanitized = slot_count_i;
    end
  end

  always_comb begin
    if (load_iterations_i == 2'd0) begin
      load_iterations_sanitized = 2'd1;
    end else if (load_iterations_i > 2'd2) begin
      load_iterations_sanitized = 2'd2;
    end else begin
      load_iterations_sanitized = load_iterations_i;
    end
  end

  always_comb begin
    if (store_burst_count_i == 2'd0) begin
      store_burst_count_sanitized = 2'd1;
    end else if ((ROWS <= 1) && (store_burst_count_i > 2'd1)) begin
      store_burst_count_sanitized = 2'd1;
    end else if (store_burst_count_i > 2'd2) begin
      store_burst_count_sanitized = 2'd2;
    end else begin
      store_burst_count_sanitized = store_burst_count_i;
    end
  end

  always_comb begin
    if (slot_stride_i == ADDR_ZERO) begin
      slot_stride_sanitized = 'd1;
    end else begin
      slot_stride_sanitized = slot_stride_i;
    end
  end

  always_comb begin
    if (store_stride_i == ADDR_ZERO) begin
      store_stride_sanitized = 'd1;
    end else begin
      store_stride_sanitized = store_stride_i;
    end
  end

  always_comb begin
    state_d = state_q;
    slot_index_d = slot_index_q;
    load_index_d = load_index_q;
    store_index_d = store_index_q;
    compute_count_d = compute_count_q;
    program_slot_count_d = program_slot_count_q;
    program_load_iterations_d = program_load_iterations_q;
    program_compute_iterations_d = program_compute_iterations_q;
    program_clear_on_done_d = program_clear_on_done_q;

    unique case (state_q)
      S_IDLE: begin
        if (start_i) begin
          state_d = S_DMA_ACT;
          slot_index_d = ADDR_ZERO;
          load_index_d = ADDR_ZERO;
          store_index_d = ADDR_ZERO;
          compute_count_d = '0;
          program_slot_count_d = slot_count_sanitized;
          program_load_iterations_d = load_iterations_sanitized;
          program_compute_iterations_d = compute_iterations_i;
          program_clear_on_done_d = clear_on_done_i;
        end
      end
      S_DMA_ACT: begin
        state_d = S_DMA_WGT;
      end
      S_DMA_WGT: begin
        if ((slot_index_q + 1'b1) < program_slot_count_q) begin
          slot_index_d = slot_index_q + 1'b1;
          state_d = S_DMA_ACT;
        end else begin
          load_index_d = ADDR_ZERO;
          store_index_d = ADDR_ZERO;
          state_d = S_LOAD;
        end
      end
      S_LOAD: begin
        if ((load_index_q + 1'b1) < program_load_iterations_q) begin
          load_index_d = load_index_q + 1'b1;
          state_d = S_LOAD;
        end else if (program_compute_iterations_q != 4'd0) begin
          compute_count_d = '0;
          store_index_d = ADDR_ZERO;
          state_d = S_COMPUTE;
        end else if (program_clear_on_done_q) begin
          state_d = S_CLEAR;
        end else begin
          state_d = S_DONE;
        end
      end
      S_COMPUTE: begin
        if ((compute_count_q + 1'b1) < program_compute_iterations_q) begin
          compute_count_d = compute_count_q + 1'b1;
          state_d = S_COMPUTE;
        end else begin
          store_index_d = ADDR_ZERO;
          state_d = S_STORE;
        end
      end
      S_STORE: begin
        if ((store_index_q + 1'b1) < store_burst_count_sanitized[ADDR_WIDTH-1:0]) begin
          store_index_d = store_index_q + 1'b1;
          state_d = S_STORE;
        end else begin
          store_index_d = ADDR_ZERO;
          state_d = S_FLUSH;
        end
      end
      S_FLUSH: begin
        if (program_clear_on_done_q) begin
          state_d = S_CLEAR;
        end else begin
          state_d = S_DONE;
        end
      end
      S_CLEAR: begin
        store_index_d = ADDR_ZERO;
        state_d = S_DONE;
      end
      S_DONE: begin
        store_index_d = ADDR_ZERO;
        state_d = S_IDLE;
      end
      default: state_d = S_IDLE;
    endcase
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state_q <= S_IDLE;
      slot_index_q <= ADDR_ZERO;
      load_index_q <= ADDR_ZERO;
      store_index_q <= ADDR_ZERO;
      compute_count_q <= '0;
      program_slot_count_q <= 2'd2;
      program_load_iterations_q <= 2'd2;
      program_compute_iterations_q <= 4'd2;
      program_clear_on_done_q <= 1'b1;
    end else begin
      state_q <= state_d;
      slot_index_q <= slot_index_d;
      load_index_q <= load_index_d;
      store_index_q <= store_index_d;
      compute_count_q <= compute_count_d;
      program_slot_count_q <= program_slot_count_d;
      program_load_iterations_q <= program_load_iterations_d;
      program_compute_iterations_q <= program_compute_iterations_d;
      program_clear_on_done_q <= program_clear_on_done_d;
    end
  end

  always_comb begin
    dma_valid_o = 1'b0;
    dma_write_weights_o = 1'b0;
    dma_addr_o = ADDR_ZERO;
    dma_payload_o = '0;
    load_vector_en_o = 1'b0;
    activation_read_addr_o = ADDR_ZERO;
    weight_read_addr_o = ADDR_ZERO;
    store_results_en_o = 1'b0;
    result_write_addr_o = ADDR_ZERO;
    store_burst_index_o = ADDR_ZERO;
    compute_en_o = 1'b0;
    flush_pipeline_o = 1'b0;
    clear_acc_o = 1'b0;
    busy_o = (state_q != S_IDLE) && (state_q != S_DONE);
    done_o = (state_q == S_DONE);
    state_o = state_q;

    unique case (state_q)
      S_DMA_ACT: begin
        dma_valid_o = 1'b1;
        dma_addr_o = descriptor_addr(activation_base_addr_i, slot_index_q, slot_stride_sanitized);
        if (slot_index_q == ADDR_ZERO) begin
          dma_payload_o[0 +: ROWS*DATA_WIDTH] = activation_slot0_i;
        end else begin
          dma_payload_o[0 +: ROWS*DATA_WIDTH] = activation_slot1_i;
        end
      end
      S_DMA_WGT: begin
        dma_valid_o = 1'b1;
        dma_write_weights_o = 1'b1;
        dma_addr_o = descriptor_addr(weight_base_addr_i, slot_index_q, slot_stride_sanitized);
        if (slot_index_q == ADDR_ZERO) begin
          dma_payload_o[0 +: COLS*DATA_WIDTH] = weight_slot0_i;
        end else begin
          dma_payload_o[0 +: COLS*DATA_WIDTH] = weight_slot1_i;
        end
      end
      S_LOAD: begin
        load_vector_en_o = 1'b1;
        if (program_slot_count_q == 2'd1) begin
          activation_read_addr_o = activation_base_addr_i;
          weight_read_addr_o = weight_base_addr_i;
        end else begin
          activation_read_addr_o = descriptor_addr(
            activation_base_addr_i,
            load_index_q,
            slot_stride_sanitized
          );
          weight_read_addr_o = descriptor_addr(
            weight_base_addr_i,
            load_index_q,
            slot_stride_sanitized
          );
        end
      end
      S_COMPUTE: begin
        compute_en_o = 1'b1;
      end
      S_STORE: begin
        store_results_en_o = 1'b1;
        result_write_addr_o = descriptor_addr(result_base_addr_i, store_index_q, store_stride_sanitized);
        store_burst_index_o = store_index_q;
      end
      S_FLUSH: begin
        flush_pipeline_o = 1'b1;
      end
      S_CLEAR: begin
        clear_acc_o = 1'b1;
      end
      default: begin
      end
    endcase
  end
endmodule
"""


def _scheduler_tb_template(operand_width: int) -> str:
    primary_case = _scheduler_sequence_case()
    short_case = _scheduler_short_sequence_case()
    return f"""module scheduler_tb;
  localparam int DATA_WIDTH = {operand_width};
  localparam int ROWS = 2;
  localparam int COLS = 2;
  localparam int DEPTH = 4;
  localparam int ADDR_WIDTH = $clog2(DEPTH);
  localparam int MAX_DIM = (ROWS > COLS) ? ROWS : COLS;
  localparam logic [3:0] S_IDLE = 4'd0;
  localparam logic [3:0] S_DMA_ACT = 4'd1;
  localparam logic [3:0] S_DMA_WGT = 4'd2;
  localparam logic [3:0] S_LOAD = 4'd3;
  localparam logic [3:0] S_COMPUTE = 4'd4;
  localparam logic [3:0] S_STORE = 4'd5;
  localparam logic [3:0] S_FLUSH = 4'd6;
  localparam logic [3:0] S_CLEAR = 4'd7;
  localparam logic [3:0] S_DONE = 4'd8;

  logic clk;
  logic rst_n;
  logic start_i;
  logic [1:0] slot_count_i;
  logic [1:0] load_iterations_i;
  logic [3:0] compute_iterations_i;
  logic [ADDR_WIDTH-1:0] activation_base_addr_i;
  logic [ADDR_WIDTH-1:0] weight_base_addr_i;
  logic [ADDR_WIDTH-1:0] result_base_addr_i;
  logic [ADDR_WIDTH-1:0] slot_stride_i;
  logic [ADDR_WIDTH-1:0] store_stride_i;
  logic [1:0] store_burst_count_i;
  logic clear_on_done_i;
  logic signed [ROWS*DATA_WIDTH-1:0] activation_slot0_i;
  logic signed [ROWS*DATA_WIDTH-1:0] activation_slot1_i;
  logic signed [COLS*DATA_WIDTH-1:0] weight_slot0_i;
  logic signed [COLS*DATA_WIDTH-1:0] weight_slot1_i;
  logic dma_valid_o;
  logic dma_write_weights_o;
  logic [ADDR_WIDTH-1:0] dma_addr_o;
  logic signed [MAX_DIM*DATA_WIDTH-1:0] dma_payload_o;
  logic load_vector_en_o;
  logic [ADDR_WIDTH-1:0] activation_read_addr_o;
  logic [ADDR_WIDTH-1:0] weight_read_addr_o;
  logic store_results_en_o;
  logic [ADDR_WIDTH-1:0] result_write_addr_o;
  logic [ADDR_WIDTH-1:0] store_burst_index_o;
  logic compute_en_o;
  logic flush_pipeline_o;
  logic clear_acc_o;
  logic busy_o;
  logic done_o;
  logic [3:0] state_o;

  scheduler #(
    .DATA_WIDTH(DATA_WIDTH),
    .ROWS(ROWS),
    .COLS(COLS),
    .DEPTH(DEPTH)
  ) dut (
    .clk(clk),
    .rst_n(rst_n),
    .start_i(start_i),
    .slot_count_i(slot_count_i),
    .load_iterations_i(load_iterations_i),
    .compute_iterations_i(compute_iterations_i),
    .activation_base_addr_i(activation_base_addr_i),
    .weight_base_addr_i(weight_base_addr_i),
    .result_base_addr_i(result_base_addr_i),
    .slot_stride_i(slot_stride_i),
    .store_stride_i(store_stride_i),
    .store_burst_count_i(store_burst_count_i),
    .clear_on_done_i(clear_on_done_i),
    .activation_slot0_i(activation_slot0_i),
    .activation_slot1_i(activation_slot1_i),
    .weight_slot0_i(weight_slot0_i),
    .weight_slot1_i(weight_slot1_i),
    .dma_valid_o(dma_valid_o),
    .dma_write_weights_o(dma_write_weights_o),
    .dma_addr_o(dma_addr_o),
    .dma_payload_o(dma_payload_o),
    .load_vector_en_o(load_vector_en_o),
    .activation_read_addr_o(activation_read_addr_o),
    .weight_read_addr_o(weight_read_addr_o),
    .store_results_en_o(store_results_en_o),
    .result_write_addr_o(result_write_addr_o),
    .store_burst_index_o(store_burst_index_o),
    .compute_en_o(compute_en_o),
    .flush_pipeline_o(flush_pipeline_o),
    .clear_acc_o(clear_acc_o),
    .busy_o(busy_o),
    .done_o(done_o),
    .state_o(state_o)
  );

  always #5 clk = ~clk;

  task automatic step_and_expect(
    input logic start_value,
    input logic [3:0] expected_state,
    input logic expected_busy,
    input logic expected_done,
    input logic expected_dma_valid,
    input logic expected_dma_write_weights,
    input integer expected_dma_addr,
    input integer expected_payload0,
    input integer expected_payload1,
    input logic expected_load_vector,
    input integer expected_activation_read_addr,
    input integer expected_weight_read_addr,
    input logic expected_store,
    input integer expected_result_write_addr,
    input integer expected_store_burst_index,
    input logic expected_compute,
    input logic expected_flush,
    input logic expected_clear
  );
    begin
      start_i = start_value;
      @(posedge clk);
      #1;
      if (state_o !== expected_state ||
          busy_o !== expected_busy ||
          done_o !== expected_done ||
          dma_valid_o !== expected_dma_valid ||
          dma_write_weights_o !== expected_dma_write_weights ||
          dma_addr_o !== expected_dma_addr[ADDR_WIDTH-1:0] ||
          $signed(dma_payload_o[0 +: DATA_WIDTH]) !== expected_payload0 ||
          $signed(dma_payload_o[DATA_WIDTH +: DATA_WIDTH]) !== expected_payload1 ||
          load_vector_en_o !== expected_load_vector ||
          activation_read_addr_o !== expected_activation_read_addr[ADDR_WIDTH-1:0] ||
          weight_read_addr_o !== expected_weight_read_addr[ADDR_WIDTH-1:0] ||
          store_results_en_o !== expected_store ||
          result_write_addr_o !== expected_result_write_addr[ADDR_WIDTH-1:0] ||
          store_burst_index_o !== expected_store_burst_index[ADDR_WIDTH-1:0] ||
          compute_en_o !== expected_compute ||
          flush_pipeline_o !== expected_flush ||
          clear_acc_o !== expected_clear) begin
        $fatal(1, "scheduler_tb failed");
      end
      start_i = 1'b0;
    end
  endtask

  initial begin
    clk = 1'b0;
    rst_n = 1'b0;
    start_i = 1'b0;
    slot_count_i = 2'd2;
    load_iterations_i = 2'd2;
    compute_iterations_i = 4'd2;
    activation_base_addr_i = '0;
    weight_base_addr_i = '0;
    result_base_addr_i = 2;
    slot_stride_i = 1;
    store_stride_i = 1;
    store_burst_count_i = 2'd2;
    clear_on_done_i = 1'b1;
    activation_slot0_i = '0;
    activation_slot1_i = '0;
    weight_slot0_i = '0;
    weight_slot1_i = '0;
    activation_slot0_i[0 +: DATA_WIDTH] = 1;
    activation_slot0_i[DATA_WIDTH +: DATA_WIDTH] = 2;
    activation_slot1_i[0 +: DATA_WIDTH] = 3;
    activation_slot1_i[DATA_WIDTH +: DATA_WIDTH] = 4;
    weight_slot0_i[0 +: DATA_WIDTH] = 5;
    weight_slot0_i[DATA_WIDTH +: DATA_WIDTH] = 6;
    weight_slot1_i[0 +: DATA_WIDTH] = 7;
    weight_slot1_i[DATA_WIDTH +: DATA_WIDTH] = 8;
    repeat (2) @(posedge clk);
    rst_n = 1'b1;

{_render_scheduler_tb_steps(primary_case)}

    slot_count_i = 2'd1;
    load_iterations_i = 2'd2;
    compute_iterations_i = 4'd1;
    activation_base_addr_i = '0;
    weight_base_addr_i = '0;
    result_base_addr_i = 2;
    slot_stride_i = 1;
    store_stride_i = 1;
    store_burst_count_i = 2'd2;
    clear_on_done_i = 1'b0;

{_render_scheduler_tb_steps(short_case)}

    $display("scheduler_tb passed");
    $finish;
  end
endmodule
"""


def _cluster_control_template(seed_tile_count: int) -> str:
    return f"""module cluster_control #(
  parameter int DEPTH = 4,
  parameter int TILE_COUNT = {seed_tile_count},
  parameter int ADDR_WIDTH = $clog2(DEPTH)
) (
  input  logic [TILE_COUNT-1:0] tile_enable_i,
  input  logic dma_valid_i,
  input  logic dma_write_weights_i,
  input  logic [ADDR_WIDTH-1:0] dma_addr_i,
  input  logic load_vector_en_i,
  input  logic [ADDR_WIDTH-1:0] activation_read_addr_i,
  input  logic [ADDR_WIDTH-1:0] weight_read_addr_i,
  input  logic store_results_en_i,
  input  logic [ADDR_WIDTH-1:0] result_write_addr_i,
  input  logic [ADDR_WIDTH-1:0] store_burst_index_i,
  input  logic compute_en_i,
  input  logic flush_pipeline_i,
  input  logic clear_acc_i,
  output logic [TILE_COUNT-1:0] tile_dma_valid_o,
  output logic dma_write_weights_o,
  output logic dma_bank_select_o,
  output logic [ADDR_WIDTH-1:0] dma_local_addr_o,
  output logic [TILE_COUNT-1:0] tile_load_vector_en_o,
  output logic activation_read_bank_select_o,
  output logic [ADDR_WIDTH-1:0] activation_local_read_addr_o,
  output logic weight_read_bank_select_o,
  output logic [ADDR_WIDTH-1:0] weight_local_read_addr_o,
  output logic [TILE_COUNT-1:0] tile_compute_en_o,
  output logic [TILE_COUNT-1:0] tile_flush_pipeline_o,
  output logic [TILE_COUNT-1:0] tile_clear_acc_o,
  output logic [TILE_COUNT-1:0] tile_store_results_en_o,
  output logic store_results_en_o,
  output logic [ADDR_WIDTH-1:0] result_write_addr_o,
  output logic [ADDR_WIDTH-1:0] store_burst_index_o
);
  assign tile_dma_valid_o = dma_valid_i ? tile_enable_i : '0;
  assign dma_write_weights_o = dma_write_weights_i;
  assign dma_bank_select_o = dma_addr_i[0];
  assign dma_local_addr_o = dma_addr_i >> 1;
  assign tile_load_vector_en_o = load_vector_en_i ? tile_enable_i : '0;
  assign activation_read_bank_select_o = activation_read_addr_i[0];
  assign activation_local_read_addr_o = activation_read_addr_i >> 1;
  assign weight_read_bank_select_o = weight_read_addr_i[0];
  assign weight_local_read_addr_o = weight_read_addr_i >> 1;
  assign tile_compute_en_o = compute_en_i ? tile_enable_i : '0;
  assign tile_flush_pipeline_o = flush_pipeline_i ? tile_enable_i : '0;
  assign tile_clear_acc_o = clear_acc_i ? tile_enable_i : '0;
  assign tile_store_results_en_o = store_results_en_i ? tile_enable_i : '0;
  assign store_results_en_o = store_results_en_i;
  assign result_write_addr_o = result_write_addr_i;
  assign store_burst_index_o = store_burst_index_i;
endmodule
"""


def _cluster_control_tb_template() -> str:
    return """module cluster_control_tb;
  localparam int DEPTH = 4;
  localparam int TILE_COUNT = 2;
  localparam int ADDR_WIDTH = $clog2(DEPTH);

  logic [TILE_COUNT-1:0] tile_enable_i;
  logic dma_valid_i;
  logic dma_write_weights_i;
  logic [ADDR_WIDTH-1:0] dma_addr_i;
  logic load_vector_en_i;
  logic [ADDR_WIDTH-1:0] activation_read_addr_i;
  logic [ADDR_WIDTH-1:0] weight_read_addr_i;
  logic store_results_en_i;
  logic [ADDR_WIDTH-1:0] result_write_addr_i;
  logic [ADDR_WIDTH-1:0] store_burst_index_i;
  logic compute_en_i;
  logic flush_pipeline_i;
  logic clear_acc_i;
  logic [TILE_COUNT-1:0] tile_dma_valid_o;
  logic dma_write_weights_o;
  logic dma_bank_select_o;
  logic [ADDR_WIDTH-1:0] dma_local_addr_o;
  logic [TILE_COUNT-1:0] tile_load_vector_en_o;
  logic activation_read_bank_select_o;
  logic [ADDR_WIDTH-1:0] activation_local_read_addr_o;
  logic weight_read_bank_select_o;
  logic [ADDR_WIDTH-1:0] weight_local_read_addr_o;
  logic [TILE_COUNT-1:0] tile_compute_en_o;
  logic [TILE_COUNT-1:0] tile_flush_pipeline_o;
  logic [TILE_COUNT-1:0] tile_clear_acc_o;
  logic [TILE_COUNT-1:0] tile_store_results_en_o;
  logic store_results_en_o;
  logic [ADDR_WIDTH-1:0] result_write_addr_o;
  logic [ADDR_WIDTH-1:0] store_burst_index_o;

  cluster_control #(
    .DEPTH(DEPTH),
    .TILE_COUNT(TILE_COUNT)
  ) dut (
    .tile_enable_i(tile_enable_i),
    .dma_valid_i(dma_valid_i),
    .dma_write_weights_i(dma_write_weights_i),
    .dma_addr_i(dma_addr_i),
    .load_vector_en_i(load_vector_en_i),
    .activation_read_addr_i(activation_read_addr_i),
    .weight_read_addr_i(weight_read_addr_i),
    .store_results_en_i(store_results_en_i),
    .result_write_addr_i(result_write_addr_i),
    .store_burst_index_i(store_burst_index_i),
    .compute_en_i(compute_en_i),
    .flush_pipeline_i(flush_pipeline_i),
    .clear_acc_i(clear_acc_i),
    .tile_dma_valid_o(tile_dma_valid_o),
    .dma_write_weights_o(dma_write_weights_o),
    .dma_bank_select_o(dma_bank_select_o),
    .dma_local_addr_o(dma_local_addr_o),
    .tile_load_vector_en_o(tile_load_vector_en_o),
    .activation_read_bank_select_o(activation_read_bank_select_o),
    .activation_local_read_addr_o(activation_local_read_addr_o),
    .weight_read_bank_select_o(weight_read_bank_select_o),
    .weight_local_read_addr_o(weight_local_read_addr_o),
    .tile_compute_en_o(tile_compute_en_o),
    .tile_flush_pipeline_o(tile_flush_pipeline_o),
    .tile_clear_acc_o(tile_clear_acc_o),
    .tile_store_results_en_o(tile_store_results_en_o),
    .store_results_en_o(store_results_en_o),
    .result_write_addr_o(result_write_addr_o),
    .store_burst_index_o(store_burst_index_o)
  );

  task automatic expect_outputs(
    input logic [TILE_COUNT-1:0] expected_tile_dma_valid,
    input logic expected_dma_write_weights,
    input logic expected_dma_bank,
    input integer expected_dma_local_addr,
    input logic [TILE_COUNT-1:0] expected_tile_load,
    input logic expected_activation_bank,
    input integer expected_activation_local_addr,
    input logic expected_weight_bank,
    input integer expected_weight_local_addr,
    input logic [TILE_COUNT-1:0] expected_tile_compute,
    input logic [TILE_COUNT-1:0] expected_tile_flush,
    input logic [TILE_COUNT-1:0] expected_tile_clear,
    input logic [TILE_COUNT-1:0] expected_tile_store,
    input logic expected_store_valid,
    input integer expected_result_write_addr,
    input integer expected_store_burst_index
  );
    begin
      #1;
      if (tile_dma_valid_o !== expected_tile_dma_valid ||
          dma_write_weights_o !== expected_dma_write_weights ||
          dma_bank_select_o !== expected_dma_bank ||
          dma_local_addr_o !== expected_dma_local_addr[ADDR_WIDTH-1:0] ||
          tile_load_vector_en_o !== expected_tile_load ||
          activation_read_bank_select_o !== expected_activation_bank ||
          activation_local_read_addr_o !== expected_activation_local_addr[ADDR_WIDTH-1:0] ||
          weight_read_bank_select_o !== expected_weight_bank ||
          weight_local_read_addr_o !== expected_weight_local_addr[ADDR_WIDTH-1:0] ||
          tile_compute_en_o !== expected_tile_compute ||
          tile_flush_pipeline_o !== expected_tile_flush ||
          tile_clear_acc_o !== expected_tile_clear ||
          tile_store_results_en_o !== expected_tile_store ||
          store_results_en_o !== expected_store_valid ||
          result_write_addr_o !== expected_result_write_addr[ADDR_WIDTH-1:0] ||
          store_burst_index_o !== expected_store_burst_index[ADDR_WIDTH-1:0]) begin
        $fatal(1, "cluster_control_tb failed");
      end
    end
  endtask

  initial begin
    tile_enable_i = '0;
    dma_valid_i = 1'b0;
    dma_write_weights_i = 1'b0;
    dma_addr_i = '0;
    load_vector_en_i = 1'b0;
    activation_read_addr_i = '0;
    weight_read_addr_i = '0;
    store_results_en_i = 1'b0;
    result_write_addr_i = '0;
    store_burst_index_i = '0;
    compute_en_i = 1'b0;
    flush_pipeline_i = 1'b0;
    clear_acc_i = 1'b0;
    expect_outputs(2'b00, 1'b0, 1'b0, 0, 2'b00, 1'b0, 0, 1'b0, 0, 2'b00, 2'b00, 2'b00, 2'b00, 1'b0, 0, 0);

    tile_enable_i = 2'b01;
    dma_valid_i = 1'b1;
    dma_write_weights_i = 1'b0;
    dma_addr_i = 2;
    expect_outputs(2'b01, 1'b0, 1'b0, 1, 2'b00, 1'b0, 0, 1'b0, 0, 2'b00, 2'b00, 2'b00, 2'b00, 1'b0, 0, 0);

    tile_enable_i = 2'b10;
    dma_write_weights_i = 1'b1;
    dma_addr_i = 3;
    load_vector_en_i = 1'b1;
    activation_read_addr_i = 2;
    weight_read_addr_i = 1;
    compute_en_i = 1'b1;
    expect_outputs(2'b10, 1'b1, 1'b1, 1, 2'b10, 1'b0, 1, 1'b1, 0, 2'b10, 2'b00, 2'b00, 2'b00, 1'b0, 0, 0);

    tile_enable_i = 2'b11;
    dma_valid_i = 1'b0;
    dma_write_weights_i = 1'b0;
    load_vector_en_i = 1'b0;
    compute_en_i = 1'b0;
    flush_pipeline_i = 1'b1;
    clear_acc_i = 1'b1;
    store_results_en_i = 1'b1;
    result_write_addr_i = 2;
    store_burst_index_i = 1;
    expect_outputs(2'b00, 1'b0, 1'b1, 1, 2'b00, 1'b0, 1, 1'b1, 0, 2'b00, 2'b11, 2'b11, 2'b11, 1'b1, 2, 1);

    $display("cluster_control_tb passed");
    $finish;
  end
endmodule
"""


def _top_npu_template(operand_width: int, acc_width: int, seed_rows: int, seed_cols: int, seed_tile_count: int) -> str:
    return f"""module top_npu #(
  parameter int DATA_WIDTH = {operand_width},
  parameter int ACC_WIDTH = {acc_width},
  parameter int ROWS = {seed_rows},
  parameter int COLS = {seed_cols},
  parameter int DEPTH = 4,
  parameter int TILE_COUNT = {seed_tile_count},
  parameter int PE_COUNT = ROWS * COLS,
  parameter int TOTAL_PE_COUNT = TILE_COUNT * PE_COUNT,
  parameter int ADDR_WIDTH = $clog2(DEPTH),
  parameter int MAX_DIM = (ROWS > COLS) ? ROWS : COLS
) (
  input  logic clk,
  input  logic rst_n,
  input  logic start_i,
  input  logic [TILE_COUNT-1:0] tile_enable_i,
  input  logic [1:0] slot_count_i,
  input  logic [1:0] load_iterations_i,
  input  logic [3:0] compute_iterations_i,
  input  logic [ADDR_WIDTH-1:0] activation_base_addr_i,
  input  logic [ADDR_WIDTH-1:0] weight_base_addr_i,
  input  logic [ADDR_WIDTH-1:0] result_base_addr_i,
  input  logic [ADDR_WIDTH-1:0] slot_stride_i,
  input  logic [ADDR_WIDTH-1:0] store_stride_i,
  input  logic [1:0] store_burst_count_i,
  input  logic clear_on_done_i,
  input  logic signed [ROWS*DATA_WIDTH-1:0] activation_slot0_i,
  input  logic signed [ROWS*DATA_WIDTH-1:0] activation_slot1_i,
  input  logic signed [COLS*DATA_WIDTH-1:0] weight_slot0_i,
  input  logic signed [COLS*DATA_WIDTH-1:0] weight_slot1_i,
  output logic busy_o,
  output logic done_o,
  output logic [3:0] scheduler_state_o,
  output logic result_write_valid_o,
  output logic [ADDR_WIDTH-1:0] result_write_addr_o,
  output logic signed [TOTAL_PE_COUNT*ACC_WIDTH-1:0] result_write_payload_o,
  output logic [TOTAL_PE_COUNT-1:0] result_write_valid_mask_o,
  output logic signed [TOTAL_PE_COUNT*ACC_WIDTH-1:0] psums_o,
  output logic [TOTAL_PE_COUNT-1:0] valids_o
);
  logic dma_valid;
  logic dma_write_weights;
  logic [ADDR_WIDTH-1:0] dma_addr;
  logic signed [MAX_DIM*DATA_WIDTH-1:0] dma_payload;
  logic load_vector_en;
  logic [ADDR_WIDTH-1:0] activation_read_addr;
  logic [ADDR_WIDTH-1:0] weight_read_addr;
  logic store_results_en;
  logic [ADDR_WIDTH-1:0] result_write_addr;
  logic [ADDR_WIDTH-1:0] store_burst_index;
  logic compute_en;
  logic flush_pipeline;
  logic clear_acc;
  logic [TILE_COUNT-1:0] tile_dma_valid;
  logic dma_write_weights_routed;
  logic dma_bank_select;
  logic [ADDR_WIDTH-1:0] dma_local_addr;
  logic [TILE_COUNT-1:0] tile_load_vector_en;
  logic activation_read_bank_select;
  logic [ADDR_WIDTH-1:0] activation_local_read_addr;
  logic weight_read_bank_select;
  logic [ADDR_WIDTH-1:0] weight_local_read_addr;
  logic [TILE_COUNT-1:0] tile_compute_en;
  logic [TILE_COUNT-1:0] tile_flush_pipeline;
  logic [TILE_COUNT-1:0] tile_clear_acc;
  logic [TILE_COUNT-1:0] tile_store_results_en;
  logic store_results_en_routed;
  logic [ADDR_WIDTH-1:0] result_write_addr_routed;
  logic [ADDR_WIDTH-1:0] store_burst_index_routed;
  logic [TILE_COUNT-1:0] scratchpad_vector_valid_unused;
  logic [TILE_COUNT-1:0] dma_done_unused;
  logic [TILE_COUNT-1:0] dma_busy_unused;
  genvar tile_idx;
  genvar store_tile_idx;
  genvar store_row_idx;
  genvar store_col_idx;

  scheduler #(
    .DATA_WIDTH(DATA_WIDTH),
    .ROWS(ROWS),
    .COLS(COLS),
    .DEPTH(DEPTH)
  ) scheduler_inst (
    .clk(clk),
    .rst_n(rst_n),
    .start_i(start_i),
    .slot_count_i(slot_count_i),
    .load_iterations_i(load_iterations_i),
    .compute_iterations_i(compute_iterations_i),
    .activation_base_addr_i(activation_base_addr_i),
    .weight_base_addr_i(weight_base_addr_i),
    .result_base_addr_i(result_base_addr_i),
    .slot_stride_i(slot_stride_i),
    .store_stride_i(store_stride_i),
    .store_burst_count_i(store_burst_count_i),
    .clear_on_done_i(clear_on_done_i),
    .activation_slot0_i(activation_slot0_i),
    .activation_slot1_i(activation_slot1_i),
    .weight_slot0_i(weight_slot0_i),
    .weight_slot1_i(weight_slot1_i),
    .dma_valid_o(dma_valid),
    .dma_write_weights_o(dma_write_weights),
    .dma_addr_o(dma_addr),
    .dma_payload_o(dma_payload),
    .load_vector_en_o(load_vector_en),
    .activation_read_addr_o(activation_read_addr),
    .weight_read_addr_o(weight_read_addr),
    .store_results_en_o(store_results_en),
    .result_write_addr_o(result_write_addr),
    .store_burst_index_o(store_burst_index),
    .compute_en_o(compute_en),
    .flush_pipeline_o(flush_pipeline),
    .clear_acc_o(clear_acc),
    .busy_o(busy_o),
    .done_o(done_o),
    .state_o(scheduler_state_o)
  );

  cluster_control #(
    .DEPTH(DEPTH),
    .TILE_COUNT(TILE_COUNT)
  ) cluster_control_inst (
    .tile_enable_i(tile_enable_i),
    .dma_valid_i(dma_valid),
    .dma_write_weights_i(dma_write_weights),
    .dma_addr_i(dma_addr),
    .load_vector_en_i(load_vector_en),
    .activation_read_addr_i(activation_read_addr),
    .weight_read_addr_i(weight_read_addr),
    .store_results_en_i(store_results_en),
    .result_write_addr_i(result_write_addr),
    .store_burst_index_i(store_burst_index),
    .compute_en_i(compute_en),
    .flush_pipeline_i(flush_pipeline),
    .clear_acc_i(clear_acc),
    .tile_dma_valid_o(tile_dma_valid),
    .dma_write_weights_o(dma_write_weights_routed),
    .dma_bank_select_o(dma_bank_select),
    .dma_local_addr_o(dma_local_addr),
    .tile_load_vector_en_o(tile_load_vector_en),
    .activation_read_bank_select_o(activation_read_bank_select),
    .activation_local_read_addr_o(activation_local_read_addr),
    .weight_read_bank_select_o(weight_read_bank_select),
    .weight_local_read_addr_o(weight_local_read_addr),
    .tile_compute_en_o(tile_compute_en),
    .tile_flush_pipeline_o(tile_flush_pipeline),
    .tile_clear_acc_o(tile_clear_acc),
    .tile_store_results_en_o(tile_store_results_en),
    .store_results_en_o(store_results_en_routed),
    .result_write_addr_o(result_write_addr_routed),
    .store_burst_index_o(store_burst_index_routed)
  );

  assign result_write_valid_o = store_results_en_routed;
  assign result_write_addr_o = result_write_addr_routed;

  generate
    for (tile_idx = 0; tile_idx < TILE_COUNT; tile_idx = tile_idx + 1) begin : gen_tiles
      tile_compute_unit #(
        .DATA_WIDTH(DATA_WIDTH),
        .ACC_WIDTH(ACC_WIDTH),
        .ROWS(ROWS),
        .COLS(COLS),
        .DEPTH(DEPTH)
      ) tile_compute_inst (
        .clk(clk),
        .rst_n(rst_n),
        .dma_valid_i(tile_dma_valid[tile_idx]),
        .dma_write_weights_i(dma_write_weights_routed),
        .dma_addr_i(dma_local_addr),
        .dma_payload_i(dma_payload),
        .activation_write_bank_i(dma_bank_select),
        .weight_write_bank_i(dma_bank_select),
        .load_vector_en_i(tile_load_vector_en[tile_idx]),
        .activation_read_bank_i(activation_read_bank_select),
        .activation_read_addr_i(activation_local_read_addr),
        .weight_read_bank_i(weight_read_bank_select),
        .weight_read_addr_i(weight_local_read_addr),
        .compute_en_i(tile_compute_en[tile_idx]),
        .flush_pipeline_i(tile_flush_pipeline[tile_idx]),
        .clear_acc_i(tile_clear_acc[tile_idx]),
        .scratchpad_vector_valid_o(scratchpad_vector_valid_unused[tile_idx]),
        .dma_done_o(dma_done_unused[tile_idx]),
        .dma_busy_o(dma_busy_unused[tile_idx]),
        .psums_o(psums_o[(tile_idx * PE_COUNT * ACC_WIDTH) +: (PE_COUNT * ACC_WIDTH)]),
        .valids_o(valids_o[(tile_idx * PE_COUNT) +: PE_COUNT])
      );
    end
    for (store_tile_idx = 0; store_tile_idx < TILE_COUNT; store_tile_idx = store_tile_idx + 1) begin : gen_store_tiles
      for (store_row_idx = 0; store_row_idx < ROWS; store_row_idx = store_row_idx + 1) begin : gen_store_rows
        for (store_col_idx = 0; store_col_idx < COLS; store_col_idx = store_col_idx + 1) begin : gen_store_cols
          localparam int STORE_LANE_IDX = (store_tile_idx * PE_COUNT) + (store_row_idx * COLS) + store_col_idx;
          localparam logic [ADDR_WIDTH-1:0] STORE_ROW_ADDR = store_row_idx[ADDR_WIDTH-1:0];
          assign result_write_valid_mask_o[STORE_LANE_IDX] =
            tile_store_results_en[store_tile_idx] &&
            (store_burst_index_routed == STORE_ROW_ADDR);
          assign result_write_payload_o[STORE_LANE_IDX*ACC_WIDTH +: ACC_WIDTH] =
            (
              tile_store_results_en[store_tile_idx] &&
              (store_burst_index_routed == STORE_ROW_ADDR)
            ) ? psums_o[STORE_LANE_IDX*ACC_WIDTH +: ACC_WIDTH] : '0;
        end
      end
    end
  endgenerate
endmodule
"""


def _top_npu_tb_template(operand_width: int, acc_width: int) -> str:
    primary_case = _top_npu_sequence_case()
    short_case = _top_npu_short_sequence_case()
    dual_tile_case = _top_npu_dual_tile_sequence_case()
    return f"""module top_npu_tb;
  localparam int DATA_WIDTH = {operand_width};
  localparam int ACC_WIDTH = {acc_width};
  localparam int ROWS = 2;
  localparam int COLS = 2;
  localparam int DEPTH = 4;
  localparam int ADDR_WIDTH = $clog2(DEPTH);
  localparam int TILE_COUNT = 2;
  localparam int PE_COUNT = ROWS * COLS;
  localparam int TOTAL_PE_COUNT = TILE_COUNT * PE_COUNT;
  localparam logic [3:0] S_IDLE = 4'd0;
  localparam logic [3:0] S_DMA_ACT = 4'd1;
  localparam logic [3:0] S_DMA_WGT = 4'd2;
  localparam logic [3:0] S_LOAD = 4'd3;
  localparam logic [3:0] S_COMPUTE = 4'd4;
  localparam logic [3:0] S_STORE = 4'd5;
  localparam logic [3:0] S_FLUSH = 4'd6;
  localparam logic [3:0] S_CLEAR = 4'd7;
  localparam logic [3:0] S_DONE = 4'd8;

  logic clk;
  logic rst_n;
  logic start_i;
  logic [TILE_COUNT-1:0] tile_enable_i;
  logic [1:0] slot_count_i;
  logic [1:0] load_iterations_i;
  logic [3:0] compute_iterations_i;
  logic [ADDR_WIDTH-1:0] activation_base_addr_i;
  logic [ADDR_WIDTH-1:0] weight_base_addr_i;
  logic [ADDR_WIDTH-1:0] result_base_addr_i;
  logic [ADDR_WIDTH-1:0] slot_stride_i;
  logic [ADDR_WIDTH-1:0] store_stride_i;
  logic [1:0] store_burst_count_i;
  logic clear_on_done_i;
  logic signed [ROWS*DATA_WIDTH-1:0] activation_slot0_i;
  logic signed [ROWS*DATA_WIDTH-1:0] activation_slot1_i;
  logic signed [COLS*DATA_WIDTH-1:0] weight_slot0_i;
  logic signed [COLS*DATA_WIDTH-1:0] weight_slot1_i;
  logic busy_o;
  logic done_o;
  logic [3:0] scheduler_state_o;
  logic result_write_valid_o;
  logic [ADDR_WIDTH-1:0] result_write_addr_o;
  logic signed [TOTAL_PE_COUNT*ACC_WIDTH-1:0] result_write_payload_o;
  logic [TOTAL_PE_COUNT-1:0] result_write_valid_mask_o;
  logic signed [TOTAL_PE_COUNT*ACC_WIDTH-1:0] psums_o;
  logic [TOTAL_PE_COUNT-1:0] valids_o;

  top_npu #(
    .DATA_WIDTH(DATA_WIDTH),
    .ACC_WIDTH(ACC_WIDTH),
    .ROWS(ROWS),
    .COLS(COLS),
    .DEPTH(DEPTH),
    .TILE_COUNT(TILE_COUNT)
  ) dut (
    .clk(clk),
    .rst_n(rst_n),
    .start_i(start_i),
    .tile_enable_i(tile_enable_i),
    .slot_count_i(slot_count_i),
    .load_iterations_i(load_iterations_i),
    .compute_iterations_i(compute_iterations_i),
    .activation_base_addr_i(activation_base_addr_i),
    .weight_base_addr_i(weight_base_addr_i),
    .result_base_addr_i(result_base_addr_i),
    .slot_stride_i(slot_stride_i),
    .store_stride_i(store_stride_i),
    .store_burst_count_i(store_burst_count_i),
    .clear_on_done_i(clear_on_done_i),
    .activation_slot0_i(activation_slot0_i),
    .activation_slot1_i(activation_slot1_i),
    .weight_slot0_i(weight_slot0_i),
    .weight_slot1_i(weight_slot1_i),
    .busy_o(busy_o),
    .done_o(done_o),
    .scheduler_state_o(scheduler_state_o),
    .result_write_valid_o(result_write_valid_o),
    .result_write_addr_o(result_write_addr_o),
    .result_write_payload_o(result_write_payload_o),
    .result_write_valid_mask_o(result_write_valid_mask_o),
    .psums_o(psums_o),
    .valids_o(valids_o)
  );

  always #5 clk = ~clk;

  task automatic step_and_expect(
    input logic start_value,
    input logic [3:0] expected_state,
    input logic expected_busy,
    input logic expected_done,
    input integer t0_p0,
    input integer t0_p1,
    input integer t0_p2,
    input integer t0_p3,
    input integer t1_p0,
    input integer t1_p1,
    input integer t1_p2,
    input integer t1_p3,
    input logic expected_store_valid,
    input integer expected_store_addr,
    input logic [TOTAL_PE_COUNT-1:0] expected_store_mask,
    input logic [TOTAL_PE_COUNT-1:0] expected_valids
  );
    begin
      start_i = start_value;
      @(posedge clk);
      #1;
      if (scheduler_state_o !== expected_state ||
          busy_o !== expected_busy ||
          done_o !== expected_done ||
          $signed(psums_o[0 +: ACC_WIDTH]) !== t0_p0 ||
          $signed(psums_o[ACC_WIDTH +: ACC_WIDTH]) !== t0_p1 ||
          $signed(psums_o[2*ACC_WIDTH +: ACC_WIDTH]) !== t0_p2 ||
          $signed(psums_o[3*ACC_WIDTH +: ACC_WIDTH]) !== t0_p3 ||
          $signed(psums_o[4*ACC_WIDTH +: ACC_WIDTH]) !== t1_p0 ||
          $signed(psums_o[5*ACC_WIDTH +: ACC_WIDTH]) !== t1_p1 ||
          $signed(psums_o[6*ACC_WIDTH +: ACC_WIDTH]) !== t1_p2 ||
          $signed(psums_o[7*ACC_WIDTH +: ACC_WIDTH]) !== t1_p3 ||
          result_write_valid_o !== expected_store_valid ||
          result_write_addr_o !== expected_store_addr[ADDR_WIDTH-1:0] ||
          result_write_valid_mask_o !== expected_store_mask ||
          (expected_store_valid &&
            ((expected_store_mask[0] && $signed(result_write_payload_o[0 +: ACC_WIDTH]) !== t0_p0) ||
             (!expected_store_mask[0] && $signed(result_write_payload_o[0 +: ACC_WIDTH]) !== 0) ||
             (expected_store_mask[1] && $signed(result_write_payload_o[ACC_WIDTH +: ACC_WIDTH]) !== t0_p1) ||
             (!expected_store_mask[1] && $signed(result_write_payload_o[ACC_WIDTH +: ACC_WIDTH]) !== 0) ||
             (expected_store_mask[2] && $signed(result_write_payload_o[2*ACC_WIDTH +: ACC_WIDTH]) !== t0_p2) ||
             (!expected_store_mask[2] && $signed(result_write_payload_o[2*ACC_WIDTH +: ACC_WIDTH]) !== 0) ||
             (expected_store_mask[3] && $signed(result_write_payload_o[3*ACC_WIDTH +: ACC_WIDTH]) !== t0_p3) ||
             (!expected_store_mask[3] && $signed(result_write_payload_o[3*ACC_WIDTH +: ACC_WIDTH]) !== 0) ||
             (expected_store_mask[4] && $signed(result_write_payload_o[4*ACC_WIDTH +: ACC_WIDTH]) !== t1_p0) ||
             (!expected_store_mask[4] && $signed(result_write_payload_o[4*ACC_WIDTH +: ACC_WIDTH]) !== 0) ||
             (expected_store_mask[5] && $signed(result_write_payload_o[5*ACC_WIDTH +: ACC_WIDTH]) !== t1_p1) ||
             (!expected_store_mask[5] && $signed(result_write_payload_o[5*ACC_WIDTH +: ACC_WIDTH]) !== 0) ||
             (expected_store_mask[6] && $signed(result_write_payload_o[6*ACC_WIDTH +: ACC_WIDTH]) !== t1_p2) ||
             (!expected_store_mask[6] && $signed(result_write_payload_o[6*ACC_WIDTH +: ACC_WIDTH]) !== 0) ||
             (expected_store_mask[7] && $signed(result_write_payload_o[7*ACC_WIDTH +: ACC_WIDTH]) !== t1_p3) ||
             (!expected_store_mask[7] && $signed(result_write_payload_o[7*ACC_WIDTH +: ACC_WIDTH]) !== 0))) ||
          valids_o !== expected_valids) begin
        $fatal(
          1,
          "top_npu_tb failed: expected state=%0d busy=%0d done=%0d store=%0d addr=%0d tile0=(%0d %0d %0d %0d) tile1=(%0d %0d %0d %0d) valids=%0b got state=%0d busy=%0d done=%0d store=%0d addr=%0d tile0=(%0d %0d %0d %0d) tile1=(%0d %0d %0d %0d) valids=%0b",
          expected_state,
          expected_busy,
          expected_done,
          expected_store_valid,
          expected_store_addr,
          t0_p0,
          t0_p1,
          t0_p2,
          t0_p3,
          t1_p0,
          t1_p1,
          t1_p2,
          t1_p3,
          expected_valids,
          scheduler_state_o,
          busy_o,
          done_o,
          result_write_valid_o,
          result_write_addr_o,
          $signed(psums_o[0 +: ACC_WIDTH]),
          $signed(psums_o[ACC_WIDTH +: ACC_WIDTH]),
          $signed(psums_o[2*ACC_WIDTH +: ACC_WIDTH]),
          $signed(psums_o[3*ACC_WIDTH +: ACC_WIDTH]),
          $signed(psums_o[4*ACC_WIDTH +: ACC_WIDTH]),
          $signed(psums_o[5*ACC_WIDTH +: ACC_WIDTH]),
          $signed(psums_o[6*ACC_WIDTH +: ACC_WIDTH]),
          $signed(psums_o[7*ACC_WIDTH +: ACC_WIDTH]),
          valids_o
        );
      end
      start_i = 1'b0;
    end
  endtask

  initial begin
    clk = 1'b0;
    rst_n = 1'b0;
    start_i = 1'b0;
    tile_enable_i = 2'b01;
    slot_count_i = 2'd2;
    load_iterations_i = 2'd2;
    compute_iterations_i = 4'd2;
    activation_base_addr_i = '0;
    weight_base_addr_i = '0;
    result_base_addr_i = 2;
    slot_stride_i = 1;
    store_stride_i = 1;
    store_burst_count_i = 2'd2;
    clear_on_done_i = 1'b1;
    activation_slot0_i = '0;
    activation_slot1_i = '0;
    weight_slot0_i = '0;
    weight_slot1_i = '0;
    activation_slot0_i[0 +: DATA_WIDTH] = 1;
    activation_slot0_i[DATA_WIDTH +: DATA_WIDTH] = 2;
    activation_slot1_i[0 +: DATA_WIDTH] = 3;
    activation_slot1_i[DATA_WIDTH +: DATA_WIDTH] = 4;
    weight_slot0_i[0 +: DATA_WIDTH] = 5;
    weight_slot0_i[DATA_WIDTH +: DATA_WIDTH] = 6;
    weight_slot1_i[0 +: DATA_WIDTH] = 7;
    weight_slot1_i[DATA_WIDTH +: DATA_WIDTH] = 8;
    repeat (2) @(posedge clk);
    rst_n = 1'b1;

{_render_top_npu_tb_steps(primary_case)}

    rst_n = 1'b0;
    tile_enable_i = 2'b01;
    repeat (2) @(posedge clk);
    rst_n = 1'b1;
    slot_count_i = 2'd1;
    load_iterations_i = 2'd2;
    compute_iterations_i = 4'd1;
    activation_base_addr_i = '0;
    weight_base_addr_i = '0;
    result_base_addr_i = 2;
    slot_stride_i = 1;
    store_stride_i = 1;
    store_burst_count_i = 2'd2;
    clear_on_done_i = 1'b0;

{_render_top_npu_tb_steps(short_case)}

    rst_n = 1'b0;
    tile_enable_i = 2'b11;
    repeat (2) @(posedge clk);
    rst_n = 1'b1;

{_render_top_npu_tb_steps(dual_tile_case)}

    $display("top_npu_tb passed");
    $finish;
  end
endmodule
"""


def _program_seed_vectors() -> Dict[str, object]:
    return {
        "activation_slot0_i": [1, 2],
        "activation_slot1_i": [3, 4],
        "weight_slot0_i": [5, 6],
        "weight_slot1_i": [7, 8],
        "tile_enable_i": [1],
        "slot_count_i": 2,
        "load_iterations_i": 2,
        "compute_iterations_i": 2,
        "activation_base_addr_i": 0,
        "weight_base_addr_i": 0,
        "result_base_addr_i": 2,
        "slot_stride_i": 1,
        "store_stride_i": 1,
        "store_burst_count_i": 2,
        "clear_on_done_i": 1,
    }


def _systolic_tile_rectangular_flush_case() -> Dict[str, object]:
    return {
        "name": "rectangular_tile_flush",
        "rows": 1,
        "cols": 3,
        "steps": [
        {
            "activations_west_i": [2],
            "weights_north_i": [3, 4, 5],
            "load_inputs_en": 1,
            "compute_en": 0,
            "clear_acc": 0,
            "flush_pipeline": 0,
            "expected": {
                "psums_o": [0, 0, 0],
                "valids_o": [0, 0, 0],
            },
        },
        {
            "activations_west_i": [6],
            "weights_north_i": [7, 8, 9],
            "load_inputs_en": 1,
            "compute_en": 0,
            "clear_acc": 0,
            "flush_pipeline": 0,
            "expected": {
                "psums_o": [0, 0, 0],
                "valids_o": [0, 0, 0],
            },
        },
        {
            "activations_west_i": [0],
            "weights_north_i": [0, 0, 0],
            "load_inputs_en": 0,
            "compute_en": 1,
            "clear_acc": 0,
            "flush_pipeline": 0,
            "expected": {
                "psums_o": [42, 16, 18],
                "valids_o": [1, 1, 1],
            },
        },
        {
            "activations_west_i": [0],
            "weights_north_i": [0, 0, 0],
            "load_inputs_en": 0,
            "compute_en": 0,
            "clear_acc": 0,
            "flush_pipeline": 1,
            "expected": {
                "psums_o": [42, 16, 18],
                "valids_o": [0, 0, 0],
            },
        },
        ],
    }


def _scheduler_expected(
    state: int,
    busy: int,
    done: int,
    dma_valid: int = 0,
    dma_write_weights: int = 0,
    dma_addr: int = 0,
    dma_payload: List[int] = None,
    load_vector_en: int = 0,
    activation_read_addr: int = 0,
    weight_read_addr: int = 0,
    store_results_en: int = 0,
    result_write_addr: int = 0,
    store_burst_index: int = 0,
    compute_en: int = 0,
    flush_pipeline: int = 0,
    clear_acc: int = 0,
) -> Dict[str, object]:
    return {
        "state_o": state,
        "busy_o": busy,
        "done_o": done,
        "dma_valid_o": dma_valid,
        "dma_write_weights_o": dma_write_weights,
        "dma_addr_o": dma_addr,
        "dma_payload_o": list(dma_payload or [0, 0]),
        "load_vector_en_o": load_vector_en,
        "activation_read_addr_o": activation_read_addr,
        "weight_read_addr_o": weight_read_addr,
        "store_results_en_o": store_results_en,
        "result_write_addr_o": result_write_addr,
        "store_burst_index_o": store_burst_index,
        "compute_en_o": compute_en,
        "flush_pipeline_o": flush_pipeline,
        "clear_acc_o": clear_acc,
    }


def _top_npu_expected(
    state: int,
    busy: int,
    done: int,
    psums: List[int],
    valids: List[int],
    store_valid: int = 0,
    store_addr: int = 0,
    store_payload: List[int] = None,
    store_valid_mask: List[int] = None,
) -> Dict[str, object]:
    payload = list(psums if store_payload is None else store_payload)
    return {
        "scheduler_state_o": state,
        "busy_o": busy,
        "done_o": done,
        "result_write_valid_o": store_valid,
        "result_write_addr_o": store_addr,
        "result_write_payload_o": payload,
        "result_write_valid_mask_o": list(store_valid_mask or [0 for _ in valids]),
        "psums_o": list(psums),
        "valids_o": list(valids),
    }


def _format_logic_literal(bits: List[int], width: int) -> str:
    padded = [int(bit) for bit in bits[:width]] + [0 for _ in range(max(0, width - len(bits)))]
    return f"{width}'b" + "".join(str(bit) for bit in reversed(padded))


def _render_systolic_tile_rect_tb_steps(case: Dict[str, object]) -> str:
    lines = []
    pe_count = int(case["rows"]) * int(case["cols"])
    for step in case["steps"]:
        activations = list(step["activations_west_i"]) + [0]
        weights = list(step["weights_north_i"]) + [0, 0, 0]
        psums = list(step["expected"]["psums_o"]) + [0 for _ in range(max(0, pe_count - len(step["expected"]["psums_o"])))]
        expected_valids = _format_logic_literal(step["expected"]["valids_o"], pe_count)
        lines.append(
            "    drive_and_expect("
            f"{int(activations[0])}, "
            f"{int(weights[0])}, "
            f"{int(weights[1])}, "
            f"{int(weights[2])}, "
            f"1'b{int(step['load_inputs_en'])}, "
            f"1'b{int(step['compute_en'])}, "
            f"1'b{int(step.get('flush_pipeline', 0))}, "
            f"1'b{int(step['clear_acc'])}, "
            f"{int(psums[0])}, "
            f"{int(psums[1])}, "
            f"{int(psums[2])}, "
            f"{expected_valids}"
            ");"
        )
    return "\n".join(lines)


def _render_scheduler_tb_steps(case: Dict[str, object]) -> str:
    lines = []
    for step in case["steps"]:
        expected = step["expected"]
        payload = list(expected["dma_payload_o"]) + [0, 0]
        lines.append(
            "    step_and_expect("
            f"1'b{int(step['start_i'])}, "
            f"4'd{int(expected['state_o'])}, "
            f"1'b{int(expected['busy_o'])}, "
            f"1'b{int(expected['done_o'])}, "
            f"1'b{int(expected['dma_valid_o'])}, "
            f"1'b{int(expected['dma_write_weights_o'])}, "
            f"{int(expected['dma_addr_o'])}, "
            f"{int(payload[0])}, "
            f"{int(payload[1])}, "
            f"1'b{int(expected['load_vector_en_o'])}, "
            f"{int(expected['activation_read_addr_o'])}, "
            f"{int(expected['weight_read_addr_o'])}, "
            f"1'b{int(expected['store_results_en_o'])}, "
            f"{int(expected['result_write_addr_o'])}, "
            f"{int(expected['store_burst_index_o'])}, "
            f"1'b{int(expected['compute_en_o'])}, "
            f"1'b{int(expected.get('flush_pipeline_o', 0))}, "
            f"1'b{int(expected['clear_acc_o'])}"
            ");"
        )
    return "\n".join(lines)


def _render_top_npu_tb_steps(case: Dict[str, object], total_pe_count: int = 8) -> str:
    lines = []
    for step in case["steps"]:
        expected = step["expected"]
        psums = list(expected["psums_o"]) + [0 for _ in range(max(0, total_pe_count - len(expected["psums_o"])))]
        store_mask = _format_logic_literal(expected.get("result_write_valid_mask_o", []), total_pe_count)
        valids = _format_logic_literal(expected["valids_o"], total_pe_count)
        lines.append(
            "    step_and_expect("
            f"1'b{int(step['start_i'])}, "
            f"4'd{int(expected['scheduler_state_o'])}, "
            f"1'b{int(expected['busy_o'])}, "
            f"1'b{int(expected['done_o'])}, "
            f"{int(psums[0])}, "
            f"{int(psums[1])}, "
            f"{int(psums[2])}, "
            f"{int(psums[3])}, "
            f"{int(psums[4])}, "
            f"{int(psums[5])}, "
            f"{int(psums[6])}, "
            f"{int(psums[7])}, "
            f"1'b{int(expected['result_write_valid_o'])}, "
            f"{int(expected['result_write_addr_o'])}, "
            f"{store_mask}, "
            f"{valids}"
            ");"
        )
    return "\n".join(lines)


def _scheduler_sequence_case() -> Dict[str, object]:
    vectors = _program_seed_vectors()
    starts = [1] + [0 for _ in range(13)]
    steps = []
    for start_value in starts:
        payload = dict(vectors)
        payload["start_i"] = start_value
        steps.append(payload)
    for payload, expected in zip(steps, scheduler_reference(steps=steps)):
        payload["expected"] = expected

    return {
        "name": "configurable_two_slot_sequence",
        "rows": 2,
        "cols": 2,
        "depth": 4,
        "steps": steps,
    }


def _scheduler_short_sequence_case() -> Dict[str, object]:
    vectors = _program_seed_vectors()
    vectors["slot_count_i"] = 1
    vectors["load_iterations_i"] = 2
    vectors["compute_iterations_i"] = 1
    vectors["clear_on_done_i"] = 0
    starts = [1] + [0 for _ in range(9)]
    steps = []
    for start_value in starts:
        payload = dict(vectors)
        payload["start_i"] = start_value
        steps.append(payload)
    for payload, expected in zip(steps, scheduler_reference(steps=steps)):
        payload["expected"] = expected

    return {
        "name": "single_slot_single_compute_no_clear",
        "rows": 2,
        "cols": 2,
        "depth": 4,
        "steps": steps,
    }


def _cluster_control_sequence_case() -> Dict[str, object]:
    vectors = _program_seed_vectors()
    vectors["slot_count_i"] = 1
    vectors["load_iterations_i"] = 1
    vectors["compute_iterations_i"] = 1
    vectors["clear_on_done_i"] = 1
    tile_masks = [[1, 0], [0, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1]]
    starts = [1] + [0 for _ in range(len(tile_masks) - 1)]
    steps = []
    for start_value, tile_mask in zip(starts, tile_masks):
        payload = dict(vectors)
        payload["start_i"] = start_value
        payload["tile_enable_i"] = list(tile_mask)
        steps.append(payload)
    for payload, expected in zip(
        steps,
        cluster_control_reference(steps=steps, rows=2, cols=2, tile_count=2),
    ):
        payload["expected"] = expected

    return {
        "name": "single_slot_control_routing",
        "tile_count": 2,
        "rows": 2,
        "cols": 2,
        "depth": 4,
        "steps": steps,
    }


def _top_npu_sequence_case() -> Dict[str, object]:
    vectors = _program_seed_vectors()
    starts = [1] + [0 for _ in range(13)]
    steps = []
    for start_value in starts:
        payload = dict(vectors)
        payload["start_i"] = start_value
        steps.append(payload)
    for payload, expected in zip(steps, top_npu_reference(steps=steps, rows=2, cols=2, depth=4, tile_count=1)):
        payload["expected"] = expected

    return {
        "name": "configurable_two_slot_program",
        "tile_count": 1,
        "rows": 2,
        "cols": 2,
        "depth": 4,
        "steps": steps,
    }


def _top_npu_short_sequence_case() -> Dict[str, object]:
    vectors = _program_seed_vectors()
    vectors["slot_count_i"] = 1
    vectors["load_iterations_i"] = 2
    vectors["compute_iterations_i"] = 1
    vectors["clear_on_done_i"] = 0
    starts = [1] + [0 for _ in range(9)]
    steps = []
    for start_value in starts:
        payload = dict(vectors)
        payload["start_i"] = start_value
        steps.append(payload)
    for payload, expected in zip(steps, top_npu_reference(steps=steps, rows=2, cols=2, depth=4, tile_count=1)):
        payload["expected"] = expected

    return {
        "name": "single_slot_single_compute_top",
        "tile_count": 1,
        "rows": 2,
        "cols": 2,
        "depth": 4,
        "steps": steps,
    }


def _top_npu_dual_tile_sequence_case() -> Dict[str, object]:
    vectors = _program_seed_vectors()
    vectors["tile_enable_i"] = [1, 1]
    vectors["slot_count_i"] = 1
    vectors["load_iterations_i"] = 2
    vectors["compute_iterations_i"] = 1
    vectors["clear_on_done_i"] = 0
    starts = [1] + [0 for _ in range(9)]
    steps = []
    for start_value in starts:
        payload = dict(vectors)
        payload["start_i"] = start_value
        steps.append(payload)
    for payload, expected in zip(steps, top_npu_reference(steps=steps, rows=2, cols=2, depth=4, tile_count=2)):
        payload["expected"] = expected

    return {
        "name": "dual_tile_broadcast_compute_top",
        "tile_count": 2,
        "rows": 2,
        "cols": 2,
        "depth": 4,
        "steps": steps,
    }


def _reference_cases() -> Dict[str, List[Dict[str, object]]]:
    return {
        "mac_unit": [
            {
                "name": "positive_accumulate",
                "inputs": {"a": 4, "b": 3, "acc_in": 2},
                "expected": {"acc_out": 14},
            },
            {
                "name": "signed_accumulate",
                "inputs": {"a": -5, "b": 6, "acc_in": 10},
                "expected": {"acc_out": -20},
            },
        ],
        "processing_element": [
            {
                "name": "compute_enabled",
                "inputs": {
                    "activation_i": 3,
                    "weight_i": 4,
                    "psum_i": 5,
                    "compute_en": 1,
                    "clear_acc": 0,
                },
                "expected": {"psum_o": 17, "valid_o": 1},
            },
            {
                "name": "bypass_psum",
                "inputs": {
                    "activation_i": -2,
                    "weight_i": 5,
                    "psum_i": 11,
                    "compute_en": 0,
                    "clear_acc": 0,
                },
                "expected": {"psum_o": 11, "valid_o": 0},
            },
            {
                "name": "clear_overrides_compute",
                "inputs": {
                    "activation_i": 7,
                    "weight_i": 8,
                    "psum_i": 99,
                    "compute_en": 1,
                    "clear_acc": 1,
                },
                "expected": {"psum_o": 0, "valid_o": 0},
            },
        ],
        "systolic_tile": [
            {
                "name": "two_wave_accumulate",
                "rows": 2,
                "cols": 2,
                "steps": [
                    {
                        "activations_west_i": [1, 2],
                        "weights_north_i": [5, 6],
                        "load_inputs_en": 1,
                        "compute_en": 0,
                        "clear_acc": 0,
                        "expected": {
                            "psums_o": [0, 0, 0, 0],
                            "valids_o": [0, 0, 0, 0],
                        },
                    },
                    {
                        "activations_west_i": [3, 4],
                        "weights_north_i": [7, 8],
                        "load_inputs_en": 1,
                        "compute_en": 0,
                        "clear_acc": 0,
                        "expected": {
                            "psums_o": [0, 0, 0, 0],
                            "valids_o": [0, 0, 0, 0],
                        },
                    },
                    {
                        "activations_west_i": [0, 0],
                        "weights_north_i": [0, 0],
                        "load_inputs_en": 0,
                        "compute_en": 1,
                        "clear_acc": 0,
                        "expected": {
                            "psums_o": [21, 8, 20, 12],
                            "valids_o": [1, 1, 1, 1],
                        },
                    },
                    {
                        "activations_west_i": [0, 0],
                        "weights_north_i": [0, 0],
                        "load_inputs_en": 0,
                        "compute_en": 1,
                        "clear_acc": 0,
                        "expected": {
                            "psums_o": [42, 16, 40, 24],
                            "valids_o": [1, 1, 1, 1],
                        },
                    },
                    {
                        "activations_west_i": [0, 0],
                        "weights_north_i": [0, 0],
                        "load_inputs_en": 0,
                        "compute_en": 0,
                        "clear_acc": 1,
                        "expected": {
                            "psums_o": [0, 0, 0, 0],
                            "valids_o": [0, 0, 0, 0],
                        },
                    },
                ],
            },
            {
                "name": "single_wave_hold",
                "rows": 2,
                "cols": 2,
                "steps": [
                    {
                        "activations_west_i": [9, 10],
                        "weights_north_i": [1, 2],
                        "load_inputs_en": 1,
                        "compute_en": 0,
                        "clear_acc": 0,
                        "expected": {
                            "psums_o": [0, 0, 0, 0],
                            "valids_o": [0, 0, 0, 0],
                        },
                    },
                    {
                        "activations_west_i": [0, 0],
                        "weights_north_i": [0, 0],
                        "load_inputs_en": 0,
                        "compute_en": 1,
                        "clear_acc": 0,
                        "expected": {
                            "psums_o": [9, 0, 0, 0],
                            "valids_o": [1, 1, 1, 1],
                        },
                    },
                    {
                        "activations_west_i": [0, 0],
                        "weights_north_i": [0, 0],
                        "load_inputs_en": 0,
                        "compute_en": 0,
                        "clear_acc": 0,
                        "expected": {
                            "psums_o": [9, 0, 0, 0],
                            "valids_o": [0, 0, 0, 0],
                        },
                    },
                ],
            },
        ],
        "dma_engine": [
            {
                "name": "activation_then_weight_transfer",
                "rows": 2,
                "cols": 2,
                "depth": 4,
                "steps": [
                    {
                        "dma_valid_i": 1,
                        "dma_write_weights_i": 0,
                        "dma_addr_i": 1,
                        "dma_payload_i": [3, 4],
                        "expected": {
                            "write_activations_en_o": 1,
                            "activation_write_addr_o": 1,
                            "activations_write_data_o": [3, 4],
                            "write_weights_en_o": 0,
                            "weight_write_addr_o": 1,
                            "weights_write_data_o": [3, 4],
                            "dma_done_o": 1,
                        },
                    },
                    {
                        "dma_valid_i": 0,
                        "dma_write_weights_i": 0,
                        "dma_addr_i": 0,
                        "dma_payload_i": [0, 0],
                        "expected": {
                            "write_activations_en_o": 0,
                            "activation_write_addr_o": 1,
                            "activations_write_data_o": [3, 4],
                            "write_weights_en_o": 0,
                            "weight_write_addr_o": 1,
                            "weights_write_data_o": [3, 4],
                            "dma_done_o": 0,
                        },
                    },
                    {
                        "dma_valid_i": 1,
                        "dma_write_weights_i": 1,
                        "dma_addr_i": 2,
                        "dma_payload_i": [7, 8],
                        "expected": {
                            "write_activations_en_o": 0,
                            "activation_write_addr_o": 2,
                            "activations_write_data_o": [7, 8],
                            "write_weights_en_o": 1,
                            "weight_write_addr_o": 2,
                            "weights_write_data_o": [7, 8],
                            "dma_done_o": 1,
                        },
                    },
                    {
                        "dma_valid_i": 0,
                        "dma_write_weights_i": 0,
                        "dma_addr_i": 0,
                        "dma_payload_i": [0, 0],
                        "expected": {
                            "write_activations_en_o": 0,
                            "activation_write_addr_o": 2,
                            "activations_write_data_o": [7, 8],
                            "write_weights_en_o": 0,
                            "weight_write_addr_o": 2,
                            "weights_write_data_o": [7, 8],
                            "dma_done_o": 0,
                        },
                    },
                ],
            }
        ],
        "scratchpad_controller": [
            {
                "name": "bank_write_and_read",
                "rows": 2,
                "cols": 2,
                "depth": 4,
                "steps": [
                    {
                        "write_activations_en_i": 1,
                        "activation_write_bank_i": 0,
                        "activation_write_addr_i": 0,
                        "activations_write_data_i": [1, 2],
                        "write_weights_en_i": 0,
                        "weight_write_bank_i": 0,
                        "weight_write_addr_i": 0,
                        "weights_write_data_i": [0, 0],
                        "load_vector_en_i": 0,
                        "activation_read_bank_i": 0,
                        "activation_read_addr_i": 0,
                        "weight_read_bank_i": 0,
                        "weight_read_addr_i": 0,
                        "expected": {
                            "activations_west_o": [0, 0],
                            "weights_north_o": [0, 0],
                            "vector_valid_o": 0,
                        },
                    },
                    {
                        "write_activations_en_i": 0,
                        "activation_write_bank_i": 0,
                        "activation_write_addr_i": 0,
                        "activations_write_data_i": [0, 0],
                        "write_weights_en_i": 1,
                        "weight_write_bank_i": 0,
                        "weight_write_addr_i": 0,
                        "weights_write_data_i": [5, 6],
                        "load_vector_en_i": 0,
                        "activation_read_bank_i": 0,
                        "activation_read_addr_i": 0,
                        "weight_read_bank_i": 0,
                        "weight_read_addr_i": 0,
                        "expected": {
                            "activations_west_o": [1, 2],
                            "weights_north_o": [0, 0],
                            "vector_valid_o": 0,
                        },
                    },
                    {
                        "write_activations_en_i": 1,
                        "activation_write_bank_i": 1,
                        "activation_write_addr_i": 0,
                        "activations_write_data_i": [9, 10],
                        "write_weights_en_i": 1,
                        "weight_write_bank_i": 1,
                        "weight_write_addr_i": 0,
                        "weights_write_data_i": [11, 12],
                        "load_vector_en_i": 0,
                        "activation_read_bank_i": 0,
                        "activation_read_addr_i": 0,
                        "weight_read_bank_i": 0,
                        "weight_read_addr_i": 0,
                        "expected": {
                            "activations_west_o": [1, 2],
                            "weights_north_o": [5, 6],
                            "vector_valid_o": 0,
                        },
                    },
                    {
                        "write_activations_en_i": 0,
                        "activation_write_bank_i": 0,
                        "activation_write_addr_i": 0,
                        "activations_write_data_i": [0, 0],
                        "write_weights_en_i": 0,
                        "weight_write_bank_i": 0,
                        "weight_write_addr_i": 0,
                        "weights_write_data_i": [0, 0],
                        "load_vector_en_i": 1,
                        "activation_read_bank_i": 0,
                        "activation_read_addr_i": 0,
                        "weight_read_bank_i": 0,
                        "weight_read_addr_i": 0,
                        "expected": {
                            "activations_west_o": [1, 2],
                            "weights_north_o": [5, 6],
                            "vector_valid_o": 1,
                        },
                    },
                    {
                        "write_activations_en_i": 0,
                        "activation_write_bank_i": 0,
                        "activation_write_addr_i": 0,
                        "activations_write_data_i": [0, 0],
                        "write_weights_en_i": 0,
                        "weight_write_bank_i": 0,
                        "weight_write_addr_i": 0,
                        "weights_write_data_i": [0, 0],
                        "load_vector_en_i": 1,
                        "activation_read_bank_i": 1,
                        "activation_read_addr_i": 0,
                        "weight_read_bank_i": 1,
                        "weight_read_addr_i": 0,
                        "expected": {
                            "activations_west_o": [9, 10],
                            "weights_north_o": [11, 12],
                            "vector_valid_o": 1,
                        },
                    },
                    {
                        "write_activations_en_i": 0,
                        "activation_write_bank_i": 0,
                        "activation_write_addr_i": 0,
                        "activations_write_data_i": [0, 0],
                        "write_weights_en_i": 0,
                        "weight_write_bank_i": 0,
                        "weight_write_addr_i": 0,
                        "weights_write_data_i": [0, 0],
                        "load_vector_en_i": 1,
                        "activation_read_bank_i": 1,
                        "activation_read_addr_i": 1,
                        "weight_read_bank_i": 1,
                        "weight_read_addr_i": 1,
                        "expected": {
                            "activations_west_o": [0, 0],
                            "weights_north_o": [0, 0],
                            "vector_valid_o": 0,
                        },
                    },
                ],
            }
        ],
        "tile_compute_unit": [
            {
                "name": "dma_feeds_banked_scratchpad_then_compute",
                "rows": 2,
                "cols": 2,
                "depth": 4,
                "steps": [
                    {
                        "dma_valid_i": 1,
                        "dma_write_weights_i": 0,
                        "dma_addr_i": 0,
                        "dma_payload_i": [1, 2],
                        "activation_write_bank_i": 0,
                        "weight_write_bank_i": 0,
                        "load_vector_en_i": 0,
                        "activation_read_bank_i": 0,
                        "activation_read_addr_i": 0,
                        "weight_read_bank_i": 0,
                        "weight_read_addr_i": 0,
                        "compute_en_i": 0,
                        "clear_acc_i": 0,
                        "expected": {
                            "psums_o": [0, 0, 0, 0],
                            "valids_o": [0, 0, 0, 0],
                            "scratchpad_vector_valid_o": 0,
                        },
                    },
                    {
                        "dma_valid_i": 0,
                        "dma_write_weights_i": 0,
                        "dma_addr_i": 0,
                        "dma_payload_i": [0, 0],
                        "activation_write_bank_i": 0,
                        "weight_write_bank_i": 0,
                        "load_vector_en_i": 0,
                        "activation_read_bank_i": 0,
                        "activation_read_addr_i": 0,
                        "weight_read_bank_i": 0,
                        "weight_read_addr_i": 0,
                        "compute_en_i": 0,
                        "clear_acc_i": 0,
                        "expected": {
                            "psums_o": [0, 0, 0, 0],
                            "valids_o": [0, 0, 0, 0],
                            "scratchpad_vector_valid_o": 0,
                        },
                    },
                    {
                        "dma_valid_i": 1,
                        "dma_write_weights_i": 1,
                        "dma_addr_i": 0,
                        "dma_payload_i": [5, 6],
                        "activation_write_bank_i": 0,
                        "weight_write_bank_i": 0,
                        "load_vector_en_i": 0,
                        "activation_read_bank_i": 0,
                        "activation_read_addr_i": 0,
                        "weight_read_bank_i": 0,
                        "weight_read_addr_i": 0,
                        "compute_en_i": 0,
                        "clear_acc_i": 0,
                        "expected": {
                            "psums_o": [0, 0, 0, 0],
                            "valids_o": [0, 0, 0, 0],
                            "scratchpad_vector_valid_o": 0,
                        },
                    },
                    {
                        "dma_valid_i": 0,
                        "dma_write_weights_i": 0,
                        "dma_addr_i": 0,
                        "dma_payload_i": [0, 0],
                        "activation_write_bank_i": 0,
                        "weight_write_bank_i": 0,
                        "load_vector_en_i": 0,
                        "activation_read_bank_i": 0,
                        "activation_read_addr_i": 0,
                        "weight_read_bank_i": 0,
                        "weight_read_addr_i": 0,
                        "compute_en_i": 0,
                        "clear_acc_i": 0,
                        "expected": {
                            "psums_o": [0, 0, 0, 0],
                            "valids_o": [0, 0, 0, 0],
                            "scratchpad_vector_valid_o": 0,
                        },
                    },
                    {
                        "dma_valid_i": 0,
                        "dma_write_weights_i": 0,
                        "dma_addr_i": 0,
                        "dma_payload_i": [0, 0],
                        "activation_write_bank_i": 0,
                        "weight_write_bank_i": 0,
                        "load_vector_en_i": 1,
                        "activation_read_bank_i": 0,
                        "activation_read_addr_i": 0,
                        "weight_read_bank_i": 0,
                        "weight_read_addr_i": 0,
                        "compute_en_i": 0,
                        "clear_acc_i": 0,
                        "expected": {
                            "psums_o": [0, 0, 0, 0],
                            "valids_o": [0, 0, 0, 0],
                            "scratchpad_vector_valid_o": 1,
                        },
                    },
                    {
                        "dma_valid_i": 1,
                        "dma_write_weights_i": 0,
                        "dma_addr_i": 0,
                        "dma_payload_i": [3, 4],
                        "activation_write_bank_i": 1,
                        "weight_write_bank_i": 0,
                        "load_vector_en_i": 0,
                        "activation_read_bank_i": 0,
                        "activation_read_addr_i": 0,
                        "weight_read_bank_i": 0,
                        "weight_read_addr_i": 0,
                        "compute_en_i": 0,
                        "clear_acc_i": 0,
                        "expected": {
                            "psums_o": [0, 0, 0, 0],
                            "valids_o": [0, 0, 0, 0],
                            "scratchpad_vector_valid_o": 0,
                        },
                    },
                    {
                        "dma_valid_i": 0,
                        "dma_write_weights_i": 0,
                        "dma_addr_i": 0,
                        "dma_payload_i": [0, 0],
                        "activation_write_bank_i": 0,
                        "weight_write_bank_i": 0,
                        "load_vector_en_i": 0,
                        "activation_read_bank_i": 0,
                        "activation_read_addr_i": 0,
                        "weight_read_bank_i": 0,
                        "weight_read_addr_i": 0,
                        "compute_en_i": 0,
                        "clear_acc_i": 0,
                        "expected": {
                            "psums_o": [0, 0, 0, 0],
                            "valids_o": [0, 0, 0, 0],
                            "scratchpad_vector_valid_o": 0,
                        },
                    },
                    {
                        "dma_valid_i": 1,
                        "dma_write_weights_i": 1,
                        "dma_addr_i": 0,
                        "dma_payload_i": [7, 8],
                        "activation_write_bank_i": 0,
                        "weight_write_bank_i": 1,
                        "load_vector_en_i": 0,
                        "activation_read_bank_i": 0,
                        "activation_read_addr_i": 0,
                        "weight_read_bank_i": 0,
                        "weight_read_addr_i": 0,
                        "compute_en_i": 0,
                        "clear_acc_i": 0,
                        "expected": {
                            "psums_o": [0, 0, 0, 0],
                            "valids_o": [0, 0, 0, 0],
                            "scratchpad_vector_valid_o": 0,
                        },
                    },
                    {
                        "dma_valid_i": 0,
                        "dma_write_weights_i": 0,
                        "dma_addr_i": 0,
                        "dma_payload_i": [0, 0],
                        "activation_write_bank_i": 0,
                        "weight_write_bank_i": 0,
                        "load_vector_en_i": 0,
                        "activation_read_bank_i": 0,
                        "activation_read_addr_i": 0,
                        "weight_read_bank_i": 0,
                        "weight_read_addr_i": 0,
                        "compute_en_i": 0,
                        "clear_acc_i": 0,
                        "expected": {
                            "psums_o": [0, 0, 0, 0],
                            "valids_o": [0, 0, 0, 0],
                            "scratchpad_vector_valid_o": 0,
                        },
                    },
                    {
                        "dma_valid_i": 0,
                        "dma_write_weights_i": 0,
                        "dma_addr_i": 0,
                        "dma_payload_i": [0, 0],
                        "activation_write_bank_i": 0,
                        "weight_write_bank_i": 0,
                        "load_vector_en_i": 1,
                        "activation_read_bank_i": 1,
                        "activation_read_addr_i": 0,
                        "weight_read_bank_i": 1,
                        "weight_read_addr_i": 0,
                        "compute_en_i": 0,
                        "clear_acc_i": 0,
                        "expected": {
                            "psums_o": [0, 0, 0, 0],
                            "valids_o": [0, 0, 0, 0],
                            "scratchpad_vector_valid_o": 1,
                        },
                    },
                    {
                        "dma_valid_i": 0,
                        "dma_write_weights_i": 0,
                        "dma_addr_i": 0,
                        "dma_payload_i": [0, 0],
                        "activation_write_bank_i": 0,
                        "weight_write_bank_i": 0,
                        "load_vector_en_i": 0,
                        "activation_read_bank_i": 1,
                        "activation_read_addr_i": 0,
                        "weight_read_addr_i": 0,
                        "weight_read_bank_i": 1,
                        "compute_en_i": 0,
                        "clear_acc_i": 0,
                        "expected": {
                            "psums_o": [0, 0, 0, 0],
                            "valids_o": [0, 0, 0, 0],
                            "scratchpad_vector_valid_o": 0,
                        },
                    },
                    {
                        "dma_valid_i": 0,
                        "dma_write_weights_i": 0,
                        "dma_addr_i": 0,
                        "dma_payload_i": [0, 0],
                        "activation_write_bank_i": 0,
                        "weight_write_bank_i": 0,
                        "load_vector_en_i": 0,
                        "activation_read_bank_i": 1,
                        "activation_read_addr_i": 0,
                        "weight_read_bank_i": 1,
                        "weight_read_addr_i": 0,
                        "compute_en_i": 1,
                        "clear_acc_i": 0,
                        "expected": {
                            "psums_o": [21, 8, 20, 12],
                            "valids_o": [1, 1, 1, 1],
                            "scratchpad_vector_valid_o": 0,
                        },
                    },
                    {
                        "dma_valid_i": 0,
                        "dma_write_weights_i": 0,
                        "dma_addr_i": 0,
                        "dma_payload_i": [0, 0],
                        "activation_write_bank_i": 0,
                        "weight_write_bank_i": 0,
                        "load_vector_en_i": 0,
                        "activation_read_bank_i": 1,
                        "activation_read_addr_i": 0,
                        "weight_read_bank_i": 1,
                        "weight_read_addr_i": 0,
                        "compute_en_i": 1,
                        "clear_acc_i": 0,
                        "expected": {
                            "psums_o": [42, 16, 40, 24],
                            "valids_o": [1, 1, 1, 1],
                            "scratchpad_vector_valid_o": 0,
                        },
                    },
                    {
                        "dma_valid_i": 0,
                        "dma_write_weights_i": 0,
                        "dma_addr_i": 0,
                        "dma_payload_i": [0, 0],
                        "activation_write_bank_i": 0,
                        "weight_write_bank_i": 0,
                        "load_vector_en_i": 0,
                        "activation_read_bank_i": 1,
                        "activation_read_addr_i": 0,
                        "weight_read_bank_i": 1,
                        "weight_read_addr_i": 0,
                        "compute_en_i": 0,
                        "clear_acc_i": 1,
                        "expected": {
                            "psums_o": [0, 0, 0, 0],
                            "valids_o": [0, 0, 0, 0],
                            "scratchpad_vector_valid_o": 0,
                        },
                    },
                ],
            }
        ],
        "scheduler": [_scheduler_sequence_case(), _scheduler_short_sequence_case()],
        "cluster_control": [_cluster_control_sequence_case()],
        "top_npu": [
            _top_npu_sequence_case(),
            _top_npu_short_sequence_case(),
            _top_npu_dual_tile_sequence_case(),
        ],
    }


def _design_intent_template(
    spec: RequirementSpec, architecture: ArchitectureCandidate
) -> str:
    seed_tile_rows, seed_tile_cols = _seed_tile_shape(
        architecture.tile_rows,
        architecture.tile_cols,
    )
    rationale = "\n".join(f"- {line}" for line in architecture.rationale)
    assumptions = "\n".join(f"- {line}" for line in spec.assumptions) or "- Nessuna"
    ambiguities = "\n".join(f"- {line}" for line in spec.ambiguities) or "- Nessuna"
    modules = "\n".join(f"- {name}" for name in architecture.modules)

    return f"""# Design Intent

## Requirement

{spec.original_text}

## Parsed Spec

- Precisione: {spec.numeric_precision}
- Throughput target: {spec.throughput_value} {spec.throughput_unit}
- Potenza: {spec.power_budget_watts} W
- Memoria disponibile: {spec.available_memory_mb} MB
- Bandwidth memoria: {spec.memory_bandwidth_gb_per_s} GB/s
- Workload: {spec.workload_type}
- Batch: {spec.batch_min}-{spec.batch_max}
- Interfacce: {", ".join(spec.interfaces)}
- Tecnologia: {spec.target_technology or "generic_5nm"}
- Frequenza: {spec.target_frequency_mhz or 1000.0} MHz

## Assunzioni

{assumptions}

## Ambiguita'

{ambiguities}

## Candidate Architecture

- Famiglia: {architecture.family}
- Tile: {architecture.tile_rows}x{architecture.tile_cols}
- Default `ROWS`x`COLS` del seed RTL: {seed_tile_rows}x{seed_tile_cols}
- Tile count: {architecture.tile_count}
- Default `TILE_COUNT` del seed RTL: {_seed_tile_count(architecture.tile_count)}
- Mesh logica: {architecture.pe_rows}x{architecture.pe_cols}
- PE count: {architecture.pe_count}
- SRAM per tile: {architecture.local_sram_kb_per_tile} KB
- Global buffer: {architecture.global_buffer_mb} MB
- Bus width: {architecture.bus_width_bits} bit
- Throughput stimato: {architecture.estimated_tops:.2f} TOPS

## Rationale

{rationale}

## Planned Modules

{modules}
"""
