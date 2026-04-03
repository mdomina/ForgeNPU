module scheduler #(
  parameter int DATA_WIDTH = 8,
  parameter int ROWS = 2,
  parameter int COLS = 2,
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
  output logic compute_en_o,
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
  localparam logic [3:0] S_CLEAR = 4'd5;
  localparam logic [3:0] S_DONE = 4'd6;
  localparam logic [ADDR_WIDTH-1:0] ADDR_ZERO = '0;

  logic [3:0] state_q;
  logic [3:0] state_d;
  logic [ADDR_WIDTH-1:0] slot_index_q;
  logic [ADDR_WIDTH-1:0] slot_index_d;
  logic [ADDR_WIDTH-1:0] load_index_q;
  logic [ADDR_WIDTH-1:0] load_index_d;
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
    state_d = state_q;
    slot_index_d = slot_index_q;
    load_index_d = load_index_q;
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
          state_d = S_LOAD;
        end
      end
      S_LOAD: begin
        if ((load_index_q + 1'b1) < program_load_iterations_q) begin
          load_index_d = load_index_q + 1'b1;
          state_d = S_LOAD;
        end else if (program_compute_iterations_q != 4'd0) begin
          compute_count_d = '0;
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
        end else if (program_clear_on_done_q) begin
          state_d = S_CLEAR;
        end else begin
          state_d = S_DONE;
        end
      end
      S_CLEAR: state_d = S_DONE;
      S_DONE: state_d = S_IDLE;
      default: state_d = S_IDLE;
    endcase
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state_q <= S_IDLE;
      slot_index_q <= ADDR_ZERO;
      load_index_q <= ADDR_ZERO;
      compute_count_q <= '0;
      program_slot_count_q <= 2'd2;
      program_load_iterations_q <= 2'd2;
      program_compute_iterations_q <= 4'd2;
      program_clear_on_done_q <= 1'b1;
    end else begin
      state_q <= state_d;
      slot_index_q <= slot_index_d;
      load_index_q <= load_index_d;
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
    compute_en_o = 1'b0;
    clear_acc_o = 1'b0;
    busy_o = (state_q != S_IDLE) && (state_q != S_DONE);
    done_o = (state_q == S_DONE);
    state_o = state_q;

    unique case (state_q)
      S_DMA_ACT: begin
        dma_valid_o = 1'b1;
        dma_addr_o = slot_index_q;
        if (slot_index_q == ADDR_ZERO) begin
          dma_payload_o[0 +: ROWS*DATA_WIDTH] = activation_slot0_i;
        end else begin
          dma_payload_o[0 +: ROWS*DATA_WIDTH] = activation_slot1_i;
        end
      end
      S_DMA_WGT: begin
        dma_valid_o = 1'b1;
        dma_write_weights_o = 1'b1;
        dma_addr_o = slot_index_q;
        if (slot_index_q == ADDR_ZERO) begin
          dma_payload_o[0 +: COLS*DATA_WIDTH] = weight_slot0_i;
        end else begin
          dma_payload_o[0 +: COLS*DATA_WIDTH] = weight_slot1_i;
        end
      end
      S_LOAD: begin
        load_vector_en_o = 1'b1;
        if (program_slot_count_q == 2'd1) begin
          activation_read_addr_o = ADDR_ZERO;
          weight_read_addr_o = ADDR_ZERO;
        end else begin
          activation_read_addr_o = load_index_q;
          weight_read_addr_o = load_index_q;
        end
      end
      S_COMPUTE: begin
        compute_en_o = 1'b1;
      end
      S_CLEAR: begin
        clear_acc_o = 1'b1;
      end
      default: begin
      end
    endcase
  end
endmodule
